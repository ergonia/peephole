//! Zero-copy account iteration over Solana's serialized instruction buffer.
//!
//! # Why This Exists
//!
//! The Solana runtime passes all account data to your program as a single contiguous buffer.
//! The standard `solana_program::entrypoint::deserialize` parses this buffer by allocating
//! `AccountInfo` structs and copying data into them. This library skips that—it gives you
//! typed views directly into the runtime's buffer, so reads and writes happen in-place.
//!
//! # Entrypoint Buffer Layout
//!
//! When your program is invoked, the runtime provides a buffer structured as:
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │ num_accounts: u64                           │
//! ├─────────────────────────────────────────────┤
//! │ Account 0                                   │
//! ├─────────────────────────────────────────────┤
//! │ Account 1                                   │
//! ├─────────────────────────────────────────────┤
//! │ ...                                         │
//! ├─────────────────────────────────────────────┤
//! │ instruction_data_len: u64                   │
//! ├─────────────────────────────────────────────┤
//! │ instruction_data: [u8]                      │
//! ├─────────────────────────────────────────────┤
//! │ program_id: Pubkey (32 bytes)               │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Account Encoding
//!
//! Each account slot starts with a marker byte:
//! - `0xFF` (NON_DUP_MARKER): A real account with full data follows
//! - `0x00-0xFE`: A duplicate—the byte value is the index of the original account
//!
//! Duplicates occur when the same account appears multiple times in the instruction.
//! The runtime only serializes the full data once; subsequent occurrences are just an index byte
//! plus 7 bytes of padding.
//!
//! # Real Account Layout
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │ NonDupAccountStatic (128 bytes)              │
//! │   - is_dup: u8 (always 0xFF)                 │
//! │   - is_signer: u8                            │
//! │   - is_writable: u8                          │
//! │   - executable: u8                           │
//! │   - original_data_len: u32                   │
//! │   - key: Pubkey (32 bytes)                   │
//! │   - owner: Pubkey (32 bytes)                 │
//! │   - lamports: u64                            │
//! │   - data_len: u64                            │
//! ├──────────────────────────────────────────────┤
//! │ data: [u8; data_len]                         │
//! ├──────────────────────────────────────────────┤
//! │ growth_buffer: [u8; MAX_PERMITTED_DATA_INCREASE] │
//! ├──────────────────────────────────────────────┤
//! │ padding to 8-byte alignment                  │
//! ├──────────────────────────────────────────────┤
//! │ rent_epoch: u64                              │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! The growth buffer (10KB) is reserved space that allows account data to grow during execution.
//! The padding ensures the next account starts 8-byte aligned.
//!
//! # Typed Slurping
//!
//! `static_slurp_typed_account::<T>()` reinterprets the account bytes directly as your struct.
//! No parsing, no copying—just a pointer cast. This is why `T` must be `Pod`: it guarantees
//! the struct has no padding bytes that could contain uninitialized memory, and that any
//! bit pattern is valid.
//!
//! The 8-byte alignment requirement exists because the runtime aligns accounts to 8 bytes.
//! If your struct isn't a multiple of 8 bytes, the next account won't be where we expect it.
//!
//! # Safety Model
//!
//! The unsafe methods assume you know the account layout. If you slurp a `TokenAccount` but
//! the actual data is a `MintAccount`, you get a valid pointer to garbage—field reads return
//! wrong values, writes corrupt data.
//!
//! The typical pattern is: verify a trusted signer first, then use unsafe methods:
//!
//! ```ignore
//! let mut iter = unsafe { AccountIterator::new_from_instruction(input) };
//!
//! // First account is always safe to read (can't be a dup, nothing to corrupt yet)
//! let (authority, iter) = unsafe { iter.known_next_full_account() };
//! if !is_authorized_signer(&authority.static_data.key) {
//!     return Err(Unauthorized);
//! }
//!
//! // Now we trust the transaction—safe to assume account types
//! let (token_account, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };
//! ```
//!
//! Debug builds verify invariants (no dups where you expect real accounts, correct sizes).
//! Release builds trust you completely.

use core::marker::PhantomData;

use crate::solana_export::constants::MAX_PERMITTED_DATA_INCREASE;
use crate::solana_export::constants::{BPF_ALIGN_OF_U128, NON_DUP_MARKER};
use crate::solana_export::pubkey::Pubkey;
use bytemuck::{Pod, Zeroable};

use crate::{assume, bytes::slurp};

/// Result of `AccountIterator::next_header()`.
pub enum NextHeader {
    /// All accounts consumed. Contains (instruction_data, program_id).
    Data(&'static mut [u8], &'static Pubkey),
    /// Real account header parsed, cursor ready for data operations.
    Header(AccountHeaderCursor<'static>),
    /// Duplicate marker—the `usize` is the index of the original account.
    Dup(usize, AccountIterator),
}

/// Parsed account header with cursor positioned at account data.
///
/// This type separates header parsing from data parsing, allowing you to:
/// - Inspect signer/metadata without assuming account type
/// - Peek at data before committing to parse
/// - Validate size before costly typed parsing
///
/// The cursor holds a reference to the 128-byte header. The data pointer is
/// computed from the header location (data always directly follows header).
/// Data slice creation is deferred until you call a parse method.
///
/// # Lifetime Safety
///
/// Peek methods return references with lifetime tied to `&self`. Since consuming
/// methods (`parse_data`, `skip`) take `self` by value, the borrow checker prevents
/// holding peek references while calling consuming methods—no aliasing possible.
pub struct AccountHeaderCursor<'a> {
    /// The 128-byte header with key, owner, lamports, etc.
    pub static_data: &'a NonDupAccountStatic,
    /// Accounts remaining after this one.
    remaining_accounts_after: usize,
}

/// Iterates over accounts in the Solana runtime's serialized buffer.
///
/// Walks the buffer, yielding each account (real or duplicate) until reaching
/// instruction data. Create with `new_from_instruction(input)` in your entrypoint.
pub struct AccountIterator {
    base_ptr: *mut u8,
    remaining_accounts: usize,
}

trait Slurper {
    fn get_account_size(size_in_data: &u64) -> u64;
    fn get_next_pointer(ptr: *mut u8) -> *mut u8;
}

struct Dynamic;

impl Slurper for Dynamic {
    #[inline]
    fn get_account_size(size_in_data: &u64) -> u64 {
        *size_in_data
    }

    #[inline]
    fn get_next_pointer(ptr: *mut u8) -> *mut u8 {
        unsafe { ptr.add(ptr.align_offset(BPF_ALIGN_OF_U128)) }
    }
}

struct Aligned;

impl Slurper for Aligned {
    #[inline]
    fn get_account_size(size_in_data: &u64) -> u64 {
        *size_in_data
    }

    #[inline]
    fn get_next_pointer(ptr: *mut u8) -> *mut u8 {
        ptr
    }
}

struct TypedSlurper<T>(PhantomData<T>);

impl<T> Slurper for TypedSlurper<T> {
    #[inline]
    fn get_account_size(size_in_data: &u64) -> u64 {
        let known_size = core::mem::size_of::<T>() as u64;
        unsafe {
            assume!(known_size == *size_in_data, "Known size is not real size");
        }

        known_size as u64
    }

    #[inline]
    fn get_next_pointer(ptr: *mut u8) -> *mut u8 {
        if core::mem::size_of::<T>() % BPF_ALIGN_OF_U128 == 0 || core::mem::size_of::<T>() == 0 {
            ptr
        } else {
            // the original base ptr is 8 byte aligned
            // the MAX_PERMITTED_DATA_INCREASE is multiple of 8 bytes
            // so we just need to add std::mem::size_of::<T>() % BPF_ALIGN_OF_U128
            // has to be a new function b/c i can't use generics in the const
            const fn adjustment<F>() -> usize {
                BPF_ALIGN_OF_U128 - core::mem::size_of::<F>() % BPF_ALIGN_OF_U128
            }
            let fast_next = unsafe { ptr.add(adjustment::<T>()) };

            #[cfg(debug_assertions)]
            {
                let true_next = unsafe { ptr.add(ptr.align_offset(BPF_ALIGN_OF_U128)) };
                assert_eq!(
                    fast_next, true_next,
                    "TypedSlurper next pointer miscalculated"
                );
            }

            fast_next
        }
    }
}

impl AccountIterator {
    /// Creates an iterator from the raw buffer passed to your program entrypoint.
    ///
    /// # Safety
    ///
    /// Only call this on data from the Solana runtime. The pointer must be valid
    /// and point to a properly formatted instruction buffer.
    #[inline]
    pub unsafe fn new_from_instruction(base: *mut u8) -> Self {
        let (num_accounts, next_ptr) = Self::read_number_of_accounts(base);
        Self::new_from_raw(next_ptr, num_accounts)
    }

    /// Reader the number of accounts from raw buffer passed to entrypoint
    ///
    /// # Safety
    ///
    /// Only call this on data from the Solana runtime. The pointer must be valid
    /// and point to a properly formatted instruction buffer.
    #[inline]
    pub unsafe fn read_number_of_accounts(base: *mut u8) -> (usize, *mut u8) {
        let (num_accounts, _next_ptr) = unsafe { slurp::<u64>(base) };
        (*num_accounts as usize, _next_ptr)
    }

    /// Creates an iterator from a raw pointer and count. Useful for resuming iteration
    /// or starting partway through the buffer.
    ///
    /// # Safety
    ///
    /// `base_ptr` must point to valid account data (or instruction data length if
    /// `remaining_accounts` is 0). `remaining_accounts` must match the actual count.
    #[inline]
    pub unsafe fn new_from_raw(base_ptr: *mut u8, remaining_accounts: usize) -> Self {
        Self {
            base_ptr,
            remaining_accounts,
        }
    }

    /// Clones the iterator. Use when you need to peek ahead or backtrack.
    ///
    /// # Safety
    ///
    /// Both iterators can mutate the same underlying buffer. Don't create aliasing
    /// mutable references to the same account data.
    #[inline]
    pub unsafe fn unsafe_clone(&self) -> Self {
        Self {
            base_ptr: self.base_ptr,
            remaining_accounts: self.remaining_accounts,
        }
    }

    /// Gets the next account, assuming it's a real account (not a duplicate marker).
    ///
    /// # Safety
    ///
    /// The next account must be real (marker byte `0xFF`). Panics in debug if it's a dup.
    #[inline]
    pub unsafe fn known_next_full_account(self) -> (NonDupAccount<'static>, AccountIterator) {
        self.known_next_header().parse_data()
    }

    /// Parses only the account header, returning a cursor for data operations.
    ///
    /// This separates header parsing from data parsing, enabling you to:
    /// - Inspect signer/metadata without assuming account type
    /// - Peek at data before committing to parse
    /// - Validate size before costly typed parsing
    ///
    /// # Safety
    ///
    /// The next account must be real (marker byte `0xFF`). Panics in debug if it's a dup.
    #[inline]
    pub unsafe fn known_next_header(self) -> AccountHeaderCursor<'static> {
        debug_assert!(self.remaining_accounts > 0);
        debug_assert_eq!(*self.base_ptr, NON_DUP_MARKER);

        let (static_data, _) = slurp::<NonDupAccountStatic>(self.base_ptr);

        AccountHeaderCursor {
            static_data,
            remaining_accounts_after: self.remaining_accounts - 1,
        }
    }

    /// Advances to the next account header, or returns instruction data if done.
    ///
    /// Unlike `next()`, this returns the header separately from the data, allowing
    /// you to inspect metadata before deciding how to parse the data.
    #[inline(always)]
    pub fn next_header(self) -> NextHeader {
        if self.remaining_accounts == 0 {
            let (slice, program_id) = self.slurp_instruction_data();
            NextHeader::Data(slice, program_id)
        } else {
            let is_dup = unsafe { *self.base_ptr };
            if is_dup == NON_DUP_MARKER {
                NextHeader::Header(unsafe { self.known_next_header() })
            } else {
                let next = unsafe { self.base_ptr.add(8) };
                NextHeader::Dup(is_dup as usize, unsafe {
                    AccountIterator::new_from_raw(next, self.remaining_accounts - 1)
                })
            }
        }
    }

    /// Like `known_next_full_account`, but also assumes the next account pointer is already
    /// 8-byte aligned (i.e., aligned to `BPF_ALIGN_OF_U128`). Skips alignment calculation
    /// for a minor speedup when you know alignment is guaranteed.
    ///
    /// # Safety
    ///
    /// Next account must be real and the next account pointer must be 8-byte aligned.
    #[inline]
    pub unsafe fn aligned_known_next_full_account(
        self,
    ) -> (NonDupAccount<'static>, AccountIterator) {
        self.known_next_header().parse_data_aligned()
    }

    /// Gets the next account, assuming it's real and its data matches type `T`'s
    /// size and alignment. Uses compile-time size for faster pointer arithmetic.
    ///
    /// # Safety
    ///
    /// Next account must be real with `data_len == size_of::<T>()` and proper alignment for `T`.
    #[inline]
    pub unsafe fn like_type_known_next_full_account<T>(
        self,
    ) -> (NonDupAccount<'static>, AccountIterator) {
        self.known_next_header().parse_data_like_type::<T>()
    }

    /// Slurp one typed account. Returns a `TypedNonDupAccount<T>` pointing directly into
    /// the runtime buffer—no allocation, no copy.
    ///
    /// # Safety
    ///
    /// - Next account must be real (not a dup marker)
    /// - Account `data_len` must equal `size_of::<T>()`
    /// - `T` must be `Pod + Zeroable` and 8-byte aligned
    ///
    /// If violated: in debug builds you get a panic, in release you get UB.
    #[inline]
    pub unsafe fn static_slurp_typed_account<T: Pod + Zeroable>(
        self,
    ) -> (&'static mut TypedNonDupAccount<T>, AccountIterator) {
        let (single_account, next) = self.static_slurp_typed_accounts::<T, 1>();
        (&mut single_account[0], next)
    }

    /// Slurp N typed accounts as an array. Single pointer cast for the whole batch.
    ///
    /// # Safety
    ///
    /// - All N accounts must be real (no dup markers)
    /// - Each account's `data_len` must equal `size_of::<T>()`
    /// - `T` must be `Pod + Zeroable` and 8-byte aligned
    ///
    /// Debug builds walk all N accounts to verify. Release builds trust you.
    #[inline]
    pub unsafe fn static_slurp_typed_accounts<T: Pod + Zeroable, const N: usize>(
        self,
    ) -> (&'static mut [TypedNonDupAccount<T>; N], AccountIterator)
    where
        [TypedNonDupAccount<T>; N]: Pod + Zeroable,
    {
        assume!(
            self.remaining_accounts >= N,
            "Too few accounts left for array slurp"
        );
        let (accounts, next) = self.slurp_typed_account::<T, N>();
        (accounts, unsafe {
            AccountIterator::new_from_raw(next, self.remaining_accounts - N)
        })
    }

    /// Gets the instruction data directly, assuming all accounts have been consumed.
    ///
    /// # Safety
    ///
    /// `remaining_accounts` must be 0. Panics in debug if accounts remain.
    #[inline]
    pub unsafe fn known_instruction_data(self) -> &'static mut [u8] {
        debug_assert_eq!(
            self.remaining_accounts, 0,
            "known_instruction_data called with {} accounts remaining",
            self.remaining_accounts
        );
        let (data, _program_iter) = self.slurp_instruction_data();
        data
    }

    /// Retrieves instruction data and program address together.
    ///
    /// This is a fast path for when you need both the instruction data and the program ID
    /// without iterating through `NextAccount`.
    ///
    /// # Safety
    ///
    /// The caller must ensure all accounts have been consumed (`remaining_accounts == 0`).
    /// The iterator's internal pointer must be positioned at the instruction data length field.
    ///
    /// In debug builds, this is verified with an assertion.
    ///
    #[inline]
    pub unsafe fn known_instruction_data_and_program_address(
        self,
    ) -> (&'static mut [u8], &'static Pubkey) {
        debug_assert_eq!(
            self.remaining_accounts, 0,
            "known_instruction_data_and_program_address called with {} accounts remaining",
            self.remaining_accounts
        );
        self.slurp_instruction_data()
    }

    /// Retrieves only the program address, skipping instruction data.
    ///
    /// This is a fast path for when you only need the program ID and don't need
    /// to read the instruction data.
    ///
    /// # Safety
    ///
    /// The caller must ensure all accounts have been consumed (`remaining_accounts == 0`).
    /// The iterator's internal pointer must be positioned at the instruction data length field.
    ///
    /// In debug builds, this is verified with an assertion.
    ///
    /// # Returns
    ///
    /// A reference to the program address (32-byte Pubkey).
    #[inline]
    pub unsafe fn known_program_address(self) -> &'static Pubkey {
        debug_assert_eq!(
            self.remaining_accounts, 0,
            "known_program_address called with {} accounts remaining",
            self.remaining_accounts
        );
        let (instruction_length, instruction_data) = slurp::<u64>(self.base_ptr);
        let program_id_ptr = instruction_data.add(*instruction_length as usize);
        &*(program_id_ptr as *const Pubkey)
    }

    #[inline]
    unsafe fn slurp_typed_account<T: Pod + Zeroable, const N: usize>(
        &self,
    ) -> (&'static mut [TypedNonDupAccount<T>; N], *mut u8)
    where
        [TypedNonDupAccount<T>; N]: Pod + Zeroable,
    {
        assume!(
            self.remaining_accounts >= N,
            "Too few accounts left for array slurp"
        );

        // static assert libs good enough to make these guarantees at compile time?
        // will either compile to nothing or explode at runtime
        assert_eq!(
            core::mem::size_of::<T>() % 8,
            0,
            "Account size must be a multiple of 8"
        );
        assert_eq!(
            core::mem::size_of::<TypedNonDupAccount<T>>() % 8,
            0,
            "Account array size must be a multiple of 8"
        );
        assert_eq!(
            core::mem::size_of::<[TypedNonDupAccount<T>; N]>() % 8,
            0,
            "Account array size must be a multiple of 8"
        );

        let (data, next_bytes) = slurp::<[TypedNonDupAccount<T>; N]>(self.base_ptr);

        #[cfg(debug_assertions)]
        {
            let mut copy_of_self = self.unsafe_clone();

            for _ in 0..N {
                match copy_of_self.next_header() {
                    NextHeader::Header(cursor) => {
                        if cursor.static_data.data_len != core::mem::size_of::<T>() as u64 {
                            panic!(
                                "Expected account size of {} but got {}",
                                core::mem::size_of::<T>() as u64,
                                cursor.static_data.data_len
                            );
                        }
                        copy_of_self = cursor.skip();
                    }
                    NextHeader::Dup(_, _) => {
                        panic!("Expected a real account, found a duplicate")
                    }
                    NextHeader::Data(_, _) => {
                        panic!("Expected an account, found instruction data")
                    }
                }
            }

            assert_eq!(next_bytes, copy_of_self.base_ptr());
        }

        (data, next_bytes)
    }

    #[inline]
    fn slurp_instruction_data(self) -> (&'static mut [u8], &'static Pubkey) {
        let (instruction_length, instruction_data) = unsafe { slurp::<u64>(self.base_ptr) };
        let data = unsafe {
            core::slice::from_raw_parts_mut(instruction_data, *instruction_length as usize)
        };
        let program_id =
            unsafe { &*(instruction_data.add(*instruction_length as usize) as *const Pubkey) };
        (data, program_id)
    }

    /// Number of accounts left to iterate.
    #[inline]
    pub fn remaining_accounts(&self) -> usize {
        self.remaining_accounts
    }

    /// Raw pointer to current position. Useful for manual pointer arithmetic or debugging.
    #[inline]
    pub fn base_ptr(&self) -> *mut u8 {
        self.base_ptr
    }
}

// ============================================================================
// AccountHeaderCursor implementation
// ============================================================================

impl<'a> AccountHeaderCursor<'a> {
    // ------------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------------

    /// Returns pointer to account data (immediately after header).
    #[inline]
    fn data_ptr(&self) -> *mut u8 {
        // SAFETY: static_data was created from a valid account buffer, and data
        // always directly follows the header in Solana's buffer layout.
        // .add(1) advances by size_of::<NonDupAccountStatic>() bytes
        unsafe { (self.static_data as *const NonDupAccountStatic).add(1) as *mut u8 }
    }

    // ------------------------------------------------------------------------
    // Size inspection (safe, no pointer arithmetic)
    // ------------------------------------------------------------------------

    /// Returns the account data length.
    #[inline]
    pub fn data_len(&self) -> u64 {
        self.static_data.data_len
    }

    /// Returns true if account data is at least `min_bytes` long.
    #[inline]
    pub fn has_min_size(&self, min_bytes: usize) -> bool {
        self.static_data.data_len >= min_bytes as u64
    }

    /// Returns true if account data is exactly `exact_bytes` long.
    #[inline]
    pub fn has_exact_size(&self, exact_bytes: usize) -> bool {
        self.static_data.data_len == exact_bytes as u64
    }

    /// Returns true if account data length matches `size_of::<T>()`.
    #[inline]
    pub fn has_size_of<T>(&self) -> bool {
        self.static_data.data_len == core::mem::size_of::<T>() as u64
    }

    // ------------------------------------------------------------------------
    // Peek methods
    // ------------------------------------------------------------------------

    /// Raw pointer to account data start.
    ///
    /// # Safety
    ///
    /// Bypasses lifetime tracking—caller must not hold this pointer when calling
    /// consuming methods (`parse_data`, `skip`).
    #[inline]
    pub unsafe fn peek_data_ptr(&self) -> *const u8 {
        self.data_ptr()
    }

    /// Mutable raw pointer to account data start.
    ///
    /// # Safety
    ///
    /// Bypasses lifetime tracking—caller must not hold this pointer when calling
    /// consuming methods (`parse_data`, `skip`).
    #[inline]
    pub unsafe fn peek_data_ptr_mut(&self) -> *mut u8 {
        self.data_ptr()
    }

    /// Peek at first `len` bytes of account data.
    ///
    /// Returns `None` if `data_len < len`.
    ///
    /// The returned slice borrows the cursor, preventing concurrent calls to
    /// consuming methods (`parse_data`, `skip`).
    #[inline]
    pub fn peek_bytes(&self, len: usize) -> Option<&[u8]> {
        if self.static_data.data_len < len as u64 {
            return None;
        }
        // SAFETY: data_ptr valid by construction, bounds checked above
        Some(unsafe { core::slice::from_raw_parts(self.data_ptr(), len) })
    }

    /// Peek at first `len` bytes of account data, mutable.
    ///
    /// Returns `None` if `data_len < len`.
    ///
    /// The returned slice borrows the cursor, preventing concurrent calls to
    /// consuming methods (`parse_data`, `skip`).
    #[inline]
    pub fn peek_bytes_mut(&mut self, len: usize) -> Option<&mut [u8]> {
        if self.static_data.data_len < len as u64 {
            return None;
        }
        // SAFETY: data_ptr valid by construction, bounds checked above
        Some(unsafe { core::slice::from_raw_parts_mut(self.data_ptr(), len) })
    }

    /// Peek at account data as type `T`.
    ///
    /// Returns `None` if `data_len != size_of::<T>()`.
    ///
    /// The returned reference borrows the cursor, preventing concurrent calls to
    /// consuming methods (`parse_data`, `skip`).
    #[inline]
    pub fn peek_as<T>(&self) -> Option<&T> {
        if self.static_data.data_len != core::mem::size_of::<T>() as u64 {
            return None;
        }
        // SAFETY: data_ptr valid by construction, size checked above
        Some(unsafe { &*(self.data_ptr() as *const T) })
    }

    /// Peek at account data as type `T`, mutable.
    ///
    /// Returns `None` if `data_len != size_of::<T>()`.
    ///
    /// The returned reference borrows the cursor, preventing concurrent calls to
    /// consuming methods (`parse_data`, `skip`).
    #[inline]
    pub fn peek_as_mut<T>(&mut self) -> Option<&mut T> {
        if self.static_data.data_len != core::mem::size_of::<T>() as u64 {
            return None;
        }
        // SAFETY: data_ptr valid by construction, size checked above
        Some(unsafe { &mut *(self.data_ptr() as *mut T) })
    }

    /// Compute the pointer to the next account without creating a data slice.
    #[inline]
    unsafe fn compute_next_ptr<S: Slurper>(&self) -> *mut u8 {
        let data_len = S::get_account_size(&self.static_data.data_len);
        let next = self.data_ptr().add(data_len as usize + MAX_PERMITTED_DATA_INCREASE);
        let next = S::get_next_pointer(next);
        // Skip rent_epoch (8 bytes)
        next.add(8)
    }

    // ------------------------------------------------------------------------
    // Consume/parse methods
    // ------------------------------------------------------------------------

    /// Skip this account entirely, returning iterator at next account.
    ///
    /// # Safety
    ///
    /// The cursor must have been created from a valid account.
    #[inline]
    pub unsafe fn skip(self) -> AccountIterator {
        let next = self.compute_next_ptr::<Dynamic>();
        AccountIterator::new_from_raw(next, self.remaining_accounts_after)
    }

    /// Parse full account data as `NonDupAccount`.
    ///
    /// # Safety
    ///
    /// The cursor must have been created from a valid account.
    #[inline]
    pub unsafe fn parse_data(self) -> (NonDupAccount<'a>, AccountIterator) {
        self.parse_data_impl::<Dynamic>()
    }

    /// Parse account, assuming next pointer is already 8-byte aligned.
    ///
    /// # Safety
    ///
    /// The cursor must have been created from a valid account, and the next
    /// account pointer must be 8-byte aligned.
    #[inline]
    pub unsafe fn parse_data_aligned(self) -> (NonDupAccount<'a>, AccountIterator) {
        self.parse_data_impl::<Aligned>()
    }

    /// Parse account using compile-time size of `T` for pointer arithmetic.
    ///
    /// # Safety
    ///
    /// The cursor must have been created from a valid account, and
    /// `data_len` must equal `size_of::<T>()`.
    #[inline]
    pub unsafe fn parse_data_like_type<T>(self) -> (NonDupAccount<'a>, AccountIterator) {
        self.parse_data_impl::<TypedSlurper<T>>()
    }

    #[inline]
    unsafe fn parse_data_impl<S: Slurper>(self) -> (NonDupAccount<'a>, AccountIterator) {
        let data_len = S::get_account_size(&self.static_data.data_len);
        let data_ptr = self.data_ptr();
        let data = core::slice::from_raw_parts_mut(data_ptr, data_len as usize);

        let next = data_ptr.add(data_len as usize + MAX_PERMITTED_DATA_INCREASE);
        let next = S::get_next_pointer(next);
        let (rent_epoch, next) = slurp::<u64>(next);

        let account = NonDupAccount {
            static_data: self.static_data,
            all_data: data,
            rent_epoch,
        };

        (account, AccountIterator::new_from_raw(next, self.remaining_accounts_after))
    }

    // ------------------------------------------------------------------------
    // Checked typed parsing
    // ------------------------------------------------------------------------

    /// Parse as type `T`, verifying size matches first.
    ///
    /// Returns `Err(self)` if `data_len != size_of::<T>()`, allowing you to
    /// try a different type or skip.
    ///
    /// # Safety
    ///
    /// The cursor must have been created from a valid account.
    #[inline]
    pub unsafe fn parse_typed_checked<T: Pod + Zeroable>(
        self,
    ) -> Result<(&'static mut TypedNonDupAccount<T>, AccountIterator), Self> {
        if self.static_data.data_len != core::mem::size_of::<T>() as u64 {
            return Err(self);
        }
        Ok(self.parse_typed_unchecked::<T>())
    }

    /// Parse as type `T`, assuming size matches.
    ///
    /// Debug builds assert `data_len == size_of::<T>()`.
    ///
    /// # Safety
    ///
    /// The cursor must have been created from a valid account, and
    /// `data_len` must equal `size_of::<T>()`.
    #[inline]
    pub unsafe fn parse_typed_unchecked<T: Pod + Zeroable>(
        self,
    ) -> (&'static mut TypedNonDupAccount<T>, AccountIterator) {
        debug_assert_eq!(
            self.static_data.data_len,
            core::mem::size_of::<T>() as u64,
            "Type size mismatch: expected {}, got {}",
            core::mem::size_of::<T>(),
            self.static_data.data_len
        );

        // Reinterpret from original base (header start), not data_ptr
        let header_ptr = (self.static_data as *const NonDupAccountStatic) as *mut u8;
        let (typed_account, _) = slurp::<TypedNonDupAccount<T>>(header_ptr);

        // Compute next pointer
        let next = self.compute_next_ptr::<TypedSlurper<T>>();

        (typed_account, AccountIterator::new_from_raw(next, self.remaining_accounts_after))
    }
}

/// The 128-byte fixed header for a real account in the serialized buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Eq, Debug)]
pub struct NonDupAccountStatic {
    // `0xFF` for real accounts, `0x00-0xFE` for duplicates (index of original).
    // We know this account is not a duplicate, so this is always 0xFF.
    // field is present to match layout.
    _is_dup: u8,
    /// Non-zero if this account signed the transaction.
    pub is_signer: u8,
    /// Non-zero if this account is writable.
    pub is_writable: u8,
    /// Non-zero if this account is executable (a program).
    pub executable: u8,
    /// Original data length before any realloc during this tx.
    pub original_data_len: u32,
    /// Account's public key.
    pub key: Pubkey,
    /// Owner program's public key.
    pub owner: Pubkey,
    /// Lamport balance.
    pub lamports: u64,
    /// Current data length.
    pub data_len: u64,
}

impl NonDupAccountStatic {
    #[inline]
    pub fn is_signer(&self) -> bool {
        self.is_signer != 0
    }

    #[inline]
    pub fn is_writable(&self) -> bool {
        self.is_writable != 0
    }

    #[inline]
    pub fn is_executable(&self) -> bool {
        self.executable != 0
    }
}

/// A real (non-duplicate) account parsed from the buffer.
#[derive(PartialEq, Eq)]
pub struct NonDupAccount<'a> {
    /// The 128-byte header with key, owner, lamports, etc.
    pub static_data: &'a NonDupAccountStatic,
    /// Account data (mutable).
    pub all_data: &'a mut [u8],
    /// Rent epoch (mostly deprecated but still in the layout).
    pub rent_epoch: &'a u64,
}

/// Account with data reinterpreted as type `T`. Memory layout:
///
/// ```text
/// [NonDupAccountStatic: 128 bytes]
/// [data: T]
/// [_buffer: 10KB growth reserve]
/// [rent_epoch: u64]
/// ```
#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(C)]
pub struct TypedNonDupAccount<T: Pod + Zeroable> {
    /// Header: is_signer, is_writable, key, owner, lamports, data_len, etc.
    pub static_data: NonDupAccountStatic,
    /// Account data as `T`.
    pub data: T,
    _buffer: [u8; MAX_PERMITTED_DATA_INCREASE],
    pub rent_epoch: u64,
}

unsafe impl<T: Pod + Zeroable> Pod for TypedNonDupAccount<T> {}
unsafe impl<T: Pod + Zeroable> Zeroable for TypedNonDupAccount<T> {}

impl NonDupAccount<'_> {
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.all_data
    }

    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.all_data.as_ptr()
    }

    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.all_data.as_mut_ptr()
    }
}

#[cfg(any(test, fuzzing))]
pub mod arbitrary_impls {
    #[allow(unused_imports)]
    use std::{eprintln, format, string::String, vec, vec::Vec};

    #[cfg(all(test, fuzzing))]
    compile_error!("fuzzing and test cannot both be true");

    use crate::solana_export::{self, pubkey_bytes,  unique_pubkey, IsAccount};

    use crate::bytes::PodUtils;

    use super::*;
    #[cfg(test)]
    use quickcheck::Arbitrary;

    #[derive(Clone, Debug)]
    pub enum TestAccount {
        /// A real account with static data, account data, and rent_epoch
        Real(NonDupAccountStatic, Vec<u8>, u64),
        Duplicate(u8),
    }

    #[cfg(test)]
    fn arbitrary_array<T: Arbitrary>(g: &mut quickcheck::Gen) -> [T; 32] {
        std::array::from_fn(|_| T::arbitrary(g))
    }

    #[cfg(test)]
    impl Arbitrary for TestAccount {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        use crate::solana_export::pubkey_from_array;

            let should_be_dup = u8::arbitrary(g) < 200;
            if should_be_dup {
                let index = u8::arbitrary(g);
                let index = index.wrapping_add((index == NON_DUP_MARKER) as u8);
                TestAccount::Duplicate(index)
            } else {
                let data = Vec::<u8>::arbitrary(g);
                let static_data = NonDupAccountStatic {
                    _is_dup: NON_DUP_MARKER,
                    is_signer: bool::arbitrary(g) as u8,
                    is_writable: bool::arbitrary(g) as u8,
                    executable: bool::arbitrary(g) as u8,
                    original_data_len: data.len() as u32,
                    key: pubkey_from_array(arbitrary_array(g)),
                    owner: pubkey_from_array(arbitrary_array(g)),
                    lamports: u64::arbitrary(g),
                    data_len: data.len() as u64,
                };
                let rent_epoch = u64::arbitrary(g);
                TestAccount::Real(static_data, data, rent_epoch)
            }
        }
    }

    pub fn create_test_account(
        is_signer: bool,
        is_writable: bool,
        data: Vec<u8>,
    ) -> (NonDupAccountStatic, Vec<u8>) {
        create_test_account_full(is_signer, is_writable, false, 100, data)
    }

    pub fn create_test_account_full(
        is_signer: bool,
        is_writable: bool,
        executable: bool,
        lamports: u64,
        data: Vec<u8>,
    ) -> (NonDupAccountStatic, Vec<u8>) {
        let account = NonDupAccountStatic {
            _is_dup: NON_DUP_MARKER,
            is_signer: is_signer as u8,
            is_writable: is_writable as u8,
            executable: executable as u8,
            original_data_len: data.len() as u32,
            key: unique_pubkey(),
            owner: unique_pubkey(),
            lamports,
            data_len: data.len() as u64,
        };
        (account, data)
    }

    /// Create a test account from metadata - includes all fuzzable fields
    pub fn create_test_account_with_meta(
        meta: &TestAccountMeta,
        data: Vec<u8>,
    ) -> (NonDupAccountStatic, Vec<u8>) {
        create_test_account_full(
            meta.is_signer,
            meta.is_writable,
            meta.executable,
            meta.lamports,
            data,
        )
    }

    /// Creates a test instruction buffer with the given accounts and instruction data.
    ///
    /// # Returns
    ///
    /// A tuple of (instruction_buffer, program_id) where program_id is the
    /// unique pubkey placed at the end of the buffer.
    pub fn create_test_instruction(
        accounts: Vec<TestAccount>,
        instruction_data: Vec<u8>,
    ) -> (Vec<u8>, Pubkey) {
        let program_id = unique_pubkey();
        let instruction =
            create_test_instruction_with_program_id(accounts, instruction_data, program_id);
        (instruction, program_id)
    }

    /// Creates a test instruction buffer with a specific program ID.
    ///
    /// Use this when you need to control the exact program ID in the buffer.
    pub fn create_test_instruction_with_program_id(
        accounts: Vec<TestAccount>,
        instruction_data: Vec<u8>,
        program_id: Pubkey,
    ) -> Vec<u8> {
        let mut instruction = Vec::new();
        let num_accounts = accounts.len() as u64;
        instruction.extend_from_slice(&num_accounts.to_le_bytes());

        for account in accounts {
            match account {
                TestAccount::Real(acc, data, rent_epoch) => {
                    instruction.extend_from_slice(acc.to_bytes());
                    instruction.extend_from_slice(&data);
                    instruction.extend_from_slice(&vec![0; MAX_PERMITTED_DATA_INCREASE]);
                    let current_length = instruction.len();

                    let padding_length =
                        (current_length as *mut u8).align_offset(BPF_ALIGN_OF_U128);
                    instruction.extend_from_slice(&vec![0; padding_length]);

                    assert_eq!(instruction.len() % BPF_ALIGN_OF_U128, 0);

                    instruction.extend_from_slice(&rent_epoch.to_le_bytes());
                }
                TestAccount::Duplicate(index) => {
                    instruction.extend_from_slice(&(index as u64).to_le_bytes());
                }
            }
        }

        let instruction_len = instruction_data.len() as u64;
        instruction.extend_from_slice(&instruction_len.to_le_bytes());
        instruction.extend_from_slice(&instruction_data);
        // add program ID (32 bytes) and padding for alignment (32 bytes)
        instruction.extend_from_slice(&pubkey_bytes(&program_id));
        instruction.extend_from_slice(&[0; 32]);
        instruction
    }

    /// Legacy helper for tests that don't need per-account rent_epoch
    pub fn create_test_instruction_with_rent_epoch(
        accounts: Vec<TestAccount>,
        instruction_data: Vec<u8>,
        rent_epoch: u64,
    ) -> Vec<u8> {
        // Convert old-style accounts to new format with uniform rent_epoch
        let accounts_with_rent: Vec<_> = accounts
            .into_iter()
            .map(|acc| match acc {
                TestAccount::Real(static_data, data, _) => {
                    TestAccount::Real(static_data, data, rent_epoch)
                }
                TestAccount::Duplicate(idx) => TestAccount::Duplicate(idx),
            })
            .collect();
        let (instruction, _program_id) =
            create_test_instruction(accounts_with_rent, instruction_data);
        instruction
    }

    // Typed account tests
    #[derive(Copy, Clone)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(C)]
    pub struct AlignedStruct {
        // 8-byte aligned
        pub a: u64,
        pub b: u64,
    }

    #[derive(Copy, Clone)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(C)]
    pub struct UnalignedStruct {
        // Not 8-byte aligned
        pub a: u16,
        pub b: u8,
    }

    #[derive(Copy, Clone)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(C)]
    pub struct EmptyStruct {} // Zero-sized

    // Test structs covering all size mod 8 remainders for alignment optimization testing
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size1Struct(pub [u8; 1]); // mod 8 = 1

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size2Struct(pub [u8; 2]); // mod 8 = 2

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size4Struct(pub [u8; 4]); // mod 8 = 4

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size5Struct(pub [u8; 5]); // mod 8 = 5

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size6Struct(pub [u8; 6]); // mod 8 = 6

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size7Struct(pub [u8; 7]); // mod 8 = 7

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size9Struct(pub [u8; 9]); // mod 8 = 1, just over 8

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size15Struct(pub [u8; 15]); // mod 8 = 7, just under 16

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(transparent)]
    pub struct Size13Struct(pub [u8; 13]); // mod 8 = 5

    pub fn create_typed_test_instruction<T: Copy>(data: &T) -> Vec<u8> {
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data as *const _ as *const u8, std::mem::size_of::<T>())
                .to_vec()
        };

        let (account, _) = create_test_account(true, true, data_bytes.clone());
        let (instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data_bytes, 0)], vec![]);
        instruction
    }
    #[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq, Eq)]
    #[repr(C)]
    pub struct QuickCheckAligned {
        pub a: u64,
        pub b: u64,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    #[repr(C)]
    pub struct QuickCheckUnaligned {
        pub a: u16,
        pub b: u8,
    }

    impl QuickCheckUnaligned {
        fn create_vec(&self) -> Vec<u8> {
            let rval = [&self.a.to_le_bytes()[..], &[self.b], &[0]].concat();

            // tricky tricky.this is why we have pod enforcement everywhere
            assert_eq!(rval.len(), std::mem::size_of::<QuickCheckUnaligned>());

            rval
        }
    }

    #[derive(Debug, Copy, Clone)]
    #[repr(C)]
    pub struct QuickCheckEmpty {}

    /// Metadata for a test account - all the fuzzable properties
    #[derive(Debug, Clone, Copy)]
    pub struct TestAccountMeta {
        pub is_signer: bool,
        pub is_writable: bool,
        pub executable: bool,
        pub lamports: u64,
        pub rent_epoch: u64,
    }

    impl TestAccountMeta {
        /// Create a simple test meta with default values
        pub fn simple(is_signer: bool, is_writable: bool) -> Self {
            Self {
                is_signer,
                is_writable,
                executable: false,
                lamports: 100,
                rent_epoch: 0,
            }
        }
    }

    #[cfg(test)]
    impl Arbitrary for Size1Struct {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            Self([u8::arbitrary(g)])
        }
    }

    #[cfg(test)]
    impl Arbitrary for Size2Struct {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            Self([u8::arbitrary(g), u8::arbitrary(g)])
        }
    }

    #[cfg(test)]
    impl Arbitrary for Size7Struct {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            Self(std::array::from_fn(|_| u8::arbitrary(g)))
        }
    }

    #[cfg(test)]
    impl Arbitrary for Size13Struct {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            Self(std::array::from_fn(|_| u8::arbitrary(g)))
        }
    }

    // ============================================================================
    // Unified random generation for quickcheck and cargo-fuzz
    // ============================================================================
    //
    // Problem: quickcheck uses `quickcheck::Arbitrary` trait, cargo-fuzz uses
    // `arbitrary::Arbitrary` trait. Both have different APIs:
    //   - quickcheck: `fn arbitrary(g: &mut Gen) -> Self` (infallible)
    //   - arbitrary: `fn arbitrary(u: &mut Unstructured) -> Result<Self>` (fallible)
    //
    // Solution: Define a `RandomSource` trait that abstracts over both, returning
    // `Result<T, Self::Error>`. For quickcheck, Error = Infallible (never type).
    // ============================================================================

    /// Trait abstracting over quickcheck::Gen and arbitrary::Unstructured.
    /// Allows sharing random generation logic between test and fuzz targets.
    #[cfg(any(test, fuzzing))]
    trait RandomSource {
        type Error;

        fn gen_u8(&mut self) -> Result<u8, Self::Error>;
        fn gen_u16(&mut self) -> Result<u16, Self::Error>;
        fn gen_u64(&mut self) -> Result<u64, Self::Error>;
        fn gen_bool(&mut self) -> Result<bool, Self::Error>;
        fn gen_usize(&mut self) -> Result<usize, Self::Error>;
        fn gen_bytes(&mut self, len: usize) -> Result<Vec<u8>, Self::Error>;
        fn gen_u64_vec(&mut self, len: usize) -> Result<Vec<u64>, Self::Error>;
    }

    #[cfg(test)]
    struct QuickCheckSource<'a>(&'a mut quickcheck::Gen);

    #[cfg(test)]
    impl RandomSource for QuickCheckSource<'_> {
        type Error = std::convert::Infallible;

        fn gen_u8(&mut self) -> Result<u8, Self::Error> {
            Ok(u8::arbitrary(self.0))
        }
        fn gen_u16(&mut self) -> Result<u16, Self::Error> {
            Ok(u16::arbitrary(self.0))
        }
        fn gen_u64(&mut self) -> Result<u64, Self::Error> {
            Ok(u64::arbitrary(self.0))
        }
        fn gen_bool(&mut self) -> Result<bool, Self::Error> {
            Ok(bool::arbitrary(self.0))
        }
        fn gen_usize(&mut self) -> Result<usize, Self::Error> {
            Ok(usize::arbitrary(self.0))
        }
        fn gen_bytes(&mut self, len: usize) -> Result<Vec<u8>, Self::Error> {
            Ok((0..len).map(|_| u8::arbitrary(self.0)).collect())
        }
        fn gen_u64_vec(&mut self, len: usize) -> Result<Vec<u64>, Self::Error> {
            Ok((0..len).map(|_| u64::arbitrary(self.0)).collect())
        }
    }

    #[cfg(fuzzing)]
    struct ArbitrarySource<'a, 'b>(&'a mut arbitrary::Unstructured<'b>);

    #[cfg(fuzzing)]
    impl RandomSource for ArbitrarySource<'_, '_> {
        type Error = arbitrary::Error;

        fn gen_u8(&mut self) -> Result<u8, Self::Error> {
            self.0.arbitrary()
        }
        fn gen_u16(&mut self) -> Result<u16, Self::Error> {
            self.0.arbitrary()
        }
        fn gen_u64(&mut self) -> Result<u64, Self::Error> {
            self.0.arbitrary()
        }
        fn gen_bool(&mut self) -> Result<bool, Self::Error> {
            self.0.arbitrary()
        }
        fn gen_usize(&mut self) -> Result<usize, Self::Error> {
            self.0.arbitrary()
        }
        fn gen_bytes(&mut self, len: usize) -> Result<Vec<u8>, Self::Error> {
            (0..len).map(|_| self.0.arbitrary()).collect()
        }
        fn gen_u64_vec(&mut self, len: usize) -> Result<Vec<u64>, Self::Error> {
            (0..len).map(|_| self.0.arbitrary()).collect()
        }
    }

    #[derive(Debug, Clone)]
    pub enum TestAccountType {
        Aligned(QuickCheckAligned, TestAccountMeta),
        Unaligned(QuickCheckUnaligned, TestAccountMeta),
        Empty(QuickCheckEmpty, TestAccountMeta),
        Untyped(Vec<u8>, TestAccountMeta),
        AlignedArray([(QuickCheckAligned, TestAccountMeta); 2]),
        AlignedArray3([(QuickCheckAligned, TestAccountMeta); 3]),
        /// Duplicate of a previous account. The u8 is a hint that will be
        /// taken modulo the number of real accounts seen so far.
        Duplicate(u8),
        /// Test like_type_known_next_full_account with 1-byte data (mod 8 = 1)
        LikeTypeSize1(Size1Struct, TestAccountMeta),
        /// Test like_type_known_next_full_account with 2-byte data (mod 8 = 2)
        LikeTypeSize2(Size2Struct, TestAccountMeta),
        /// Test like_type_known_next_full_account with 7-byte data (mod 8 = 7)
        LikeTypeSize7(Size7Struct, TestAccountMeta),
        /// Test like_type_known_next_full_account with 13-byte data (mod 8 = 5)
        LikeTypeSize13(Size13Struct, TestAccountMeta),
        /// Test aligned_known_next_full_account with 8-byte aligned data.
        /// Uses Vec<u64> to guarantee 8-byte aligned length - flattened to bytes when creating account.
        AlignedKnown(Vec<u64>, TestAccountMeta),
    }

    /// Shared generation logic for TestAccountType.
    /// Used by both quickcheck and arbitrary implementations.
    #[cfg(any(test, fuzzing))]
    impl TestAccountType {
        fn generate<R: RandomSource>(rng: &mut R) -> Result<Self, R::Error> {
            let variant = rng.gen_u8()? % 13;
            match variant {
                0 => {
                    let a = rng.gen_u64()?;
                    let b = rng.gen_u64()?;
                    let meta = TestAccountMeta::generate(rng)?;
                    Ok(TestAccountType::Aligned(QuickCheckAligned { a, b }, meta))
                }
                1 => {
                    let a = rng.gen_u16()?;
                    let b = rng.gen_u8()?;
                    let meta = TestAccountMeta::generate(rng)?;
                    Ok(TestAccountType::Unaligned(
                        QuickCheckUnaligned { a, b },
                        meta,
                    ))
                }
                2 => {
                    let meta = TestAccountMeta::generate(rng)?;
                    Ok(TestAccountType::Empty(QuickCheckEmpty {}, meta))
                }
                3 => {
                    let len = rng.gen_usize()? % 64;
                    let data = rng.gen_bytes(len)?;
                    let meta = TestAccountMeta::generate(rng)?;
                    Ok(TestAccountType::Untyped(data, meta))
                }
                4 => {
                    let p1 = Self::generate_aligned_pair(rng)?;
                    let p2 = Self::generate_aligned_pair(rng)?;
                    Ok(TestAccountType::AlignedArray([p1, p2]))
                }
                5 => {
                    let p1 = Self::generate_aligned_pair(rng)?;
                    let p2 = Self::generate_aligned_pair(rng)?;
                    let p3 = Self::generate_aligned_pair(rng)?;
                    Ok(TestAccountType::AlignedArray3([p1, p2, p3]))
                }
                6 => {
                    let b = rng.gen_u8()?;
                    let meta = TestAccountMeta::generate(rng)?;
                    Ok(TestAccountType::LikeTypeSize1(Size1Struct([b]), meta))
                }
                7 => {
                    let b1 = rng.gen_u8()?;
                    let b2 = rng.gen_u8()?;
                    let meta = TestAccountMeta::generate(rng)?;
                    Ok(TestAccountType::LikeTypeSize2(Size2Struct([b1, b2]), meta))
                }
                8 => {
                    let bytes = rng.gen_bytes(7)?;
                    let meta = TestAccountMeta::generate(rng)?;
                    let arr: [u8; 7] = bytes.try_into().unwrap();
                    Ok(TestAccountType::LikeTypeSize7(Size7Struct(arr), meta))
                }
                9 => {
                    let bytes = rng.gen_bytes(13)?;
                    let meta = TestAccountMeta::generate(rng)?;
                    let arr: [u8; 13] = bytes.try_into().unwrap();
                    Ok(TestAccountType::LikeTypeSize13(Size13Struct(arr), meta))
                }
                10 => {
                    let len = (rng.gen_usize()? % 4) + 1; // 1-4 u64s = 8-32 bytes
                    let data = rng.gen_u64_vec(len)?;
                    let meta = TestAccountMeta::generate(rng)?;
                    Ok(TestAccountType::AlignedKnown(data, meta))
                }
                // Remaining cases: Duplicate (~15% chance)
                _ => {
                    let hint = rng.gen_u8()?;
                    Ok(TestAccountType::Duplicate(hint))
                }
            }
        }

        fn generate_aligned_pair<R: RandomSource>(
            rng: &mut R,
        ) -> Result<(QuickCheckAligned, TestAccountMeta), R::Error> {
            let a = rng.gen_u64()?;
            let b = rng.gen_u64()?;
            let meta = TestAccountMeta::generate(rng)?;
            Ok((QuickCheckAligned { a, b }, meta))
        }
    }

    #[cfg(any(test, fuzzing))]
    impl TestAccountMeta {
        fn generate<R: RandomSource>(rng: &mut R) -> Result<Self, R::Error> {
            Ok(TestAccountMeta {
                is_signer: rng.gen_bool()?,
                is_writable: rng.gen_bool()?,
                executable: rng.gen_bool()?,
                lamports: rng.gen_u64()?,
                rent_epoch: rng.gen_u64()?,
            })
        }
    }

    #[cfg(test)]
    impl Arbitrary for TestAccountType {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut source = QuickCheckSource(g);
            // Infallible can never happen, so unwrap is safe
            TestAccountType::generate(&mut source).unwrap()
        }
    }

    #[cfg(fuzzing)]
    impl<'a> arbitrary::Arbitrary<'a> for TestAccountType {
        fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
            let mut source = ArbitrarySource(u);
            TestAccountType::generate(&mut source)
        }
    }

    #[cfg(test)]
    impl Arbitrary for TestAccountMeta {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut source = QuickCheckSource(g);
            TestAccountMeta::generate(&mut source).unwrap()
        }
    }

    #[cfg(fuzzing)]
    impl<'a> arbitrary::Arbitrary<'a> for TestAccountMeta {
        fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
            let mut source = ArbitrarySource(u);
            TestAccountMeta::generate(&mut source)
        }
    }

    /// Creates test accounts from a TestAccountType, returning (static_data, data, rent_epoch) tuples
    pub fn create_account_from_type(
        account_type: TestAccountType,
    ) -> Vec<(NonDupAccountStatic, Vec<u8>, u64)> {
        match account_type {
            TestAccountType::Aligned(aligned, meta) => {
                let data = aligned.to_vec();
                let (acc, data) = create_test_account_with_meta(&meta, data);
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::Unaligned(unaligned, meta) => {
                let data = unaligned.create_vec();
                let (acc, data) = create_test_account_with_meta(&meta, data);
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::Empty(_, meta) => {
                let (acc, data) = create_test_account_with_meta(&meta, vec![]);
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::Untyped(data, meta) => {
                let (acc, data) = create_test_account_with_meta(&meta, data.clone());
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::AlignedArray(array) => array
                .iter()
                .map(|(aligned, meta)| {
                    let data = aligned.to_vec();
                    let (acc, data) = create_test_account_with_meta(meta, data);
                    (acc, data, meta.rent_epoch)
                })
                .collect(),
            TestAccountType::AlignedArray3(array) => array
                .iter()
                .map(|(aligned, meta)| {
                    let data = aligned.to_vec();
                    let (acc, data) = create_test_account_with_meta(meta, data);
                    (acc, data, meta.rent_epoch)
                })
                .collect(),
            // Duplicates are handled separately in create_test_accounts_from_types
            TestAccountType::Duplicate(_) => vec![],
            TestAccountType::LikeTypeSize1(sized, meta) => {
                let data = sized.0.to_vec();
                let (acc, data) = create_test_account_with_meta(&meta, data);
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::LikeTypeSize2(sized, meta) => {
                let data = sized.0.to_vec();
                let (acc, data) = create_test_account_with_meta(&meta, data);
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::LikeTypeSize7(sized, meta) => {
                let data = sized.0.to_vec();
                let (acc, data) = create_test_account_with_meta(&meta, data);
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::LikeTypeSize13(sized, meta) => {
                let data = sized.0.to_vec();
                let (acc, data) = create_test_account_with_meta(&meta, data);
                vec![(acc, data, meta.rent_epoch)]
            }
            TestAccountType::AlignedKnown(u64s, meta) => {
                // Flatten Vec<u64> to bytes (little-endian)
                let data: Vec<u8> = u64s.iter().flat_map(|v| v.to_le_bytes()).collect();
                let (acc, data) = create_test_account_with_meta(&meta, data);
                vec![(acc, data, meta.rent_epoch)]
            }
        }
    }

    /// Helper to verify account metadata matches expected values.
    /// Returns false if any field doesn't match.
    fn verify_account_meta(static_data: &NonDupAccountStatic, meta: &TestAccountMeta) -> bool {
        (static_data.is_signer == 1) == meta.is_signer
            && (static_data.is_writable == 1) == meta.is_writable
            && (static_data.executable == 1) == meta.executable
            && static_data.lamports == meta.lamports
    }

    /// Helper to verify account metadata and rent_epoch for NonDupAccount.
    /// Returns false if any field doesn't match.
    fn verify_non_dup_account_meta(acc: &NonDupAccount, meta: &TestAccountMeta) -> bool {
        verify_account_meta(acc.static_data, meta) && *acc.rent_epoch == meta.rent_epoch
    }

    /// Helper to verify account metadata and rent_epoch for TypedNonDupAccount.
    /// Returns false if any field doesn't match.
    fn verify_typed_account_meta<T: Pod + Zeroable>(
        acc: &TypedNonDupAccount<T>,
        meta: &TestAccountMeta,
    ) -> bool {
        verify_account_meta(&acc.static_data, meta) && acc.rent_epoch == meta.rent_epoch
    }

    /// Converts a list of TestAccountType into a flat list of TestAccount,
    /// properly handling duplicates by computing valid indices based on
    /// how many real accounts have been seen so far.
    ///
    /// Note: The first account can never be a duplicate (invalid in Solana).
    /// Duplicate hints are taken modulo the number of real accounts seen so far.
    pub fn create_test_accounts_from_types(account_types: &[TestAccountType]) -> Vec<TestAccount> {
        let mut result = Vec::new();
        let mut real_account_count = 0usize;

        for account_type in account_types {
            match account_type {
                TestAccountType::Duplicate(hint) => {
                    // Only create a dup if there's at least one real account to reference.
                    // This naturally prevents the first account from being a duplicate.
                    if real_account_count > 0 {
                        let dup_index = (*hint as usize) % real_account_count;
                        result.push(TestAccount::Duplicate(dup_index as u8));
                    }
                    // If no real accounts yet, skip this dup (can't reference anything)
                }
                other => {
                    let real_accounts = create_account_from_type(other.clone());
                    for (acc, data, rent_epoch) in real_accounts {
                        result.push(TestAccount::Real(acc, data, rent_epoch));
                        real_account_count += 1;
                    }
                }
            }
        }

        result
    }

    pub fn do_quickcheck_mixed_account_types_with_arrays(
        account_types: Vec<TestAccountType>,
        instruction_data_gen: Vec<u8>,
    ) -> bool {
        // Filter out Duplicate variants since typed slurping doesn't support them.
        // Duplicate handling is tested separately in do_quickcheck_compare_with_solana_deserialize.
        let account_types: Vec<_> = account_types
            .into_iter()
            .filter(|t| !matches!(t, TestAccountType::Duplicate(_)))
            .collect();

        let accounts: Vec<_> = account_types
            .iter()
            .flat_map(|t| {
                let accounts = create_account_from_type(t.clone());
                accounts
                    .into_iter()
                    .map(|(acc, data, rent_epoch)| TestAccount::Real(acc, data, rent_epoch))
            })
            .collect();

        let (mut instruction, _program_id) =
            create_test_instruction(accounts, instruction_data_gen.clone());
        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        for account_type in account_types {
            match account_type {
                TestAccountType::Aligned(al, meta) => {
                    let (acc, next) =
                        unsafe { iterator.static_slurp_typed_account::<QuickCheckAligned>() };
                    if acc.static_data.data_len != std::mem::size_of::<QuickCheckAligned>() as u64
                        || !verify_typed_account_meta(acc, &meta)
                        || acc.data != al
                    {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::Unaligned(un, meta) => {
                    let (acc, next) = unsafe {
                        iterator.like_type_known_next_full_account::<QuickCheckUnaligned>()
                    };
                    if acc.static_data.data_len != std::mem::size_of::<QuickCheckUnaligned>() as u64
                        || !verify_non_dup_account_meta(&acc, &meta)
                    {
                        return false;
                    }
                    let data_as_struct =
                        unsafe { &*(acc.data().as_ptr() as *const QuickCheckUnaligned) };
                    if data_as_struct != &un {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::Empty(_, meta) => {
                    let (acc, next) =
                        unsafe { iterator.like_type_known_next_full_account::<QuickCheckEmpty>() };
                    if acc.static_data.data_len != 0 || !verify_non_dup_account_meta(&acc, &meta) {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::Untyped(data, meta) => {
                    let (acc, next) = unsafe { iterator.known_next_full_account() };
                    if acc.data() != data.as_slice() || !verify_non_dup_account_meta(&acc, &meta) {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::AlignedArray(array) => {
                    let (accs, next) =
                        unsafe { iterator.static_slurp_typed_accounts::<QuickCheckAligned, 2>() };
                    for ((given, meta), acc) in array.iter().zip(accs.iter()) {
                        if acc.static_data.data_len
                            != std::mem::size_of::<QuickCheckAligned>() as u64
                            || !verify_typed_account_meta(acc, meta)
                            || &acc.data != given
                        {
                            return false;
                        }
                    }
                    iterator = next;
                }
                TestAccountType::AlignedArray3(array) => {
                    let (accs, next) =
                        unsafe { iterator.static_slurp_typed_accounts::<QuickCheckAligned, 3>() };
                    for ((given, meta), acc) in array.iter().zip(accs.iter()) {
                        if acc.static_data.data_len
                            != std::mem::size_of::<QuickCheckAligned>() as u64
                            || !verify_typed_account_meta(acc, meta)
                            || &acc.data != given
                        {
                            return false;
                        }
                    }
                    iterator = next;
                }
                TestAccountType::Duplicate(_) => {
                    // Duplicates are filtered out before this loop, so this is unreachable
                    unreachable!("Duplicate variants should have been filtered out")
                }
                TestAccountType::LikeTypeSize1(expected, meta) => {
                    let (acc, next) =
                        unsafe { iterator.like_type_known_next_full_account::<Size1Struct>() };
                    if acc.static_data.data_len != std::mem::size_of::<Size1Struct>() as u64
                        || !verify_non_dup_account_meta(&acc, &meta)
                        || acc.data() != expected.0.as_slice()
                    {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::LikeTypeSize2(expected, meta) => {
                    let (acc, next) =
                        unsafe { iterator.like_type_known_next_full_account::<Size2Struct>() };
                    if acc.static_data.data_len != std::mem::size_of::<Size2Struct>() as u64
                        || !verify_non_dup_account_meta(&acc, &meta)
                        || acc.data() != expected.0.as_slice()
                    {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::LikeTypeSize7(expected, meta) => {
                    let (acc, next) =
                        unsafe { iterator.like_type_known_next_full_account::<Size7Struct>() };
                    if acc.static_data.data_len != std::mem::size_of::<Size7Struct>() as u64
                        || !verify_non_dup_account_meta(&acc, &meta)
                        || acc.data() != expected.0.as_slice()
                    {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::LikeTypeSize13(expected, meta) => {
                    let (acc, next) =
                        unsafe { iterator.like_type_known_next_full_account::<Size13Struct>() };
                    if acc.static_data.data_len != std::mem::size_of::<Size13Struct>() as u64
                        || !verify_non_dup_account_meta(&acc, &meta)
                        || acc.data() != expected.0.as_slice()
                    {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::AlignedKnown(expected_u64s, meta) => {
                    let (acc, next) = unsafe { iterator.aligned_known_next_full_account() };
                    // Flatten Vec<u64> to bytes for comparison
                    let expected_bytes: Vec<u8> =
                        expected_u64s.iter().flat_map(|v| v.to_le_bytes()).collect();
                    if acc.static_data.data_len != expected_bytes.len() as u64
                        || !verify_non_dup_account_meta(&acc, &meta)
                        || acc.data() != expected_bytes.as_slice()
                    {
                        return false;
                    }
                    iterator = next;
                }
            }
        }

        // Verify we've reached the instruction data
        let NextHeader::Data(check, _program_iter) = iterator.next_header() else {
            return false;
        };
        check == instruction_data_gen.as_slice()
    }

    /// Parsed account for comparison in tests.
    enum ParsedAccountForTest<'a> {
        Real(NonDupAccount<'a>),
        Dup(usize),
    }

    pub fn do_quickcheck_compare_with_solana_deserialize(
        account_types: Vec<TestAccountType>,
        instruction_data_gen: Vec<u8>,
    ) -> bool {
        // 1. Generate TestAccount structures from TestAccountType (including dups)
        let test_accounts: Vec<_> = create_test_accounts_from_types(&account_types)
            .into_iter()
            // Truncate to 256 accounts to match pinocchio entrypoint limit
            .take(256)
            .collect();

        // 2. Create the instruction buffer
        let (mut instruction_buffer, expected_program_id) =
            create_test_instruction(test_accounts.clone(), instruction_data_gen.clone());

        // 3. Parse with AccountIterator
        let mut fast_results: Vec<ParsedAccountForTest> = Vec::new();
        let mut fast_iter =
            unsafe { AccountIterator::new_from_instruction(instruction_buffer.as_mut_ptr()) };
        let (final_fast_data, fast_program_id) = loop {
            match fast_iter.next_header() {
                NextHeader::Header(cursor) => {
                    let (acc, next_iter) = unsafe { cursor.parse_data() };
                    fast_results.push(ParsedAccountForTest::Real(acc));
                    fast_iter = next_iter;
                }
                NextHeader::Dup(idx, next_iter) => {
                    fast_results.push(ParsedAccountForTest::Dup(idx));
                    fast_iter = next_iter;
                }
                NextHeader::Data(data, program_iter) => {
                    break (data.to_vec(), *program_iter); // Clone data for comparison
                }
            }
        };

        // 4. Parse with solana_program::entrypoint::deserialize
        let (program_id_solana, accounts_solana, instruction_data_solana) =
            solana_export::easy_deserialize(instruction_buffer.as_mut_ptr());

        // 5. Compare results

        // Compare instruction data
        if instruction_data_solana != final_fast_data.as_slice() {
            eprintln!(
                "Instruction data mismatch: Solana={:?}, Fast={:?}",
                instruction_data_solana, final_fast_data
            );
            return false;
        }

        // Compare number of accounts
        if fast_results.len() != accounts_solana.len() {
            eprintln!(
                "Account count mismatch: Solana={}, Fast={}",
                accounts_solana.len(),
                fast_results.len()
            );
            return false;
        }

        // Compare each account
        for (i, (fast_acc, solana_acc)) in
            fast_results.iter().zip(accounts_solana.iter()).enumerate()
        {
            match fast_acc {
                ParsedAccountForTest::Real(fast_real) => {
                    // Compare fields for real accounts
                    if fast_real.static_data.key != *solana_acc.get_key() {
                        eprintln!("Key mismatch at index {}", i);
                        return false;
                    }
                    if (fast_real.static_data.is_signer != 0) != solana_acc.get_is_signer() {
                        eprintln!("is_signer mismatch at index {}", i);
                        return false;
                    }
                    if (fast_real.static_data.is_writable != 0) != solana_acc.get_is_writable() {
                        eprintln!("is_writable mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.owner != *solana_acc.get_owner() {
                        eprintln!("Owner mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.lamports != solana_acc.get_lamports() {
                        eprintln!("Lamports mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.data_len != solana_acc.get_data_len() as u64 {
                        eprintln!("Data length mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.data() != solana_acc.get_data() {
                        eprintln!("Data mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.executable != solana_acc.get_executable() as u8 {
                        eprintln!("Executable mismatch at index {}", i);
                        return false;
                    }
                }
                ParsedAccountForTest::Dup(dup_index) => {
                    // For duplicates, Solana gives us a full AccountInfo that should
                    // have the same key as the original account at dup_index.
                    // The data also points to the same underlying buffer.
                    let original_solana_acc = &accounts_solana[*dup_index];

                    // Verify the duplicate has the same key as the original
                    if solana_acc.get_key() != original_solana_acc.get_key() {
                        eprintln!(
                            "Dup key mismatch at index {}: expected key of account {}, got different key",
                            i, dup_index
                        );
                        return false;
                    }

                    // Verify owner matches
                    if solana_acc.get_owner() != original_solana_acc.get_owner() {
                        eprintln!("Dup owner mismatch at index {}", i);
                        return false;
                    }

                    // Verify data matches (same underlying buffer)
                    if solana_acc.get_data() != original_solana_acc.get_data() {
                        eprintln!("Dup data mismatch at index {}", i);
                        return false;
                    }

                    // Verify lamports match
                    if solana_acc.get_lamports() != original_solana_acc.get_lamports() {
                        eprintln!("Dup lamports mismatch at index {}", i);
                        return false;
                    }
                }
            }
        }

        // Compare program IDs
        if fast_program_id != program_id_solana {
            eprintln!(
                "Program ID mismatch: Solana={:?}, Fast={:?}",
                program_id_solana, fast_program_id
            );
            return false;
        }

        // Verify our program ID matches the expected one we generated
        if fast_program_id != expected_program_id {
            eprintln!(
                "Program ID doesn't match expected: Expected={:?}, Got={:?}",
                expected_program_id, fast_program_id
            );
            return false;
        }

        true // All checks passed
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use std::{eprintln, format, string::String, vec, vec::Vec};

    use super::*;

    use super::arbitrary_impls::*;
    use crate::bytes::PodUtils;

    fn test_account_parsing(accounts: Vec<TestAccount>, instruction_data: Vec<u8>) {
        let (mut instruction, _program_id) =
            create_test_instruction(accounts.clone(), instruction_data.clone());
        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        for expected_account in accounts {
            match iterator.next_header() {
                NextHeader::Header(cursor) => {
                    let (parsed, next_iter) = unsafe { cursor.parse_data() };
                    match expected_account {
                        TestAccount::Real(expected, expected_data, expected_rent_epoch) => {
                            assert_eq!(parsed.static_data.is_signer, expected.is_signer);
                            assert_eq!(parsed.static_data.is_writable, expected.is_writable);
                            assert_eq!(parsed.static_data.executable, expected.executable);
                            assert_eq!(parsed.static_data.data_len, expected.data_len);
                            assert_eq!(parsed.static_data.key, expected.key);
                            assert_eq!(parsed.static_data.owner, expected.owner);
                            assert_eq!(parsed.static_data.lamports, expected.lamports);
                            assert_eq!(parsed.data(), expected_data);
                            assert_eq!(*parsed.rent_epoch, expected_rent_epoch);
                        }
                        TestAccount::Duplicate(_) => panic!("Expected real account, got dup marker"),
                    }
                    iterator = next_iter;
                }
                NextHeader::Dup(parsed_index, next_iter) => {
                    match expected_account {
                        TestAccount::Duplicate(expected_index) => {
                            assert_eq!(parsed_index, expected_index as usize);
                        }
                        TestAccount::Real(_, _, _) => panic!("Expected dup, got real account"),
                    }
                    iterator = next_iter;
                }
                NextHeader::Data(_, _) => panic!("Expected an account, found instruction data"),
            }
        }

        // Check instruction data
        match iterator.next_header() {
            NextHeader::Data(parsed_data, _) => {
                assert_eq!(parsed_data, instruction_data.as_slice())
            }
            _ => panic!("Expected instruction data"),
        }
    }

    #[quickcheck_macros::quickcheck]
    fn quickcheck_many_account_combos(test_accounts: Vec<TestAccount>, instruction_data: Vec<u8>) {
        test_account_parsing(test_accounts, instruction_data);
    }

    #[test]
    fn test_various_account_combinations() {
        let (account1, data1) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (account2, data2) = create_test_account(false, true, vec![5, 6, 7, 8, 9, 10]);
        let (account3, data3) = create_test_account(true, true, vec![11, 12]);

        let test_cases = vec![
            (vec![], vec![1, 2, 3, 4]),
            (
                vec![TestAccount::Real(account1, data1.clone(), 0)],
                vec![5, 6],
            ),
            (
                vec![
                    TestAccount::Real(account1, data1.clone(), 0),
                    TestAccount::Real(account2, data2.clone(), 0),
                ],
                vec![7, 8, 9],
            ),
            (
                vec![
                    TestAccount::Real(account1, data1.clone(), 0),
                    TestAccount::Duplicate(0),
                    TestAccount::Real(account2, data2.clone(), 0),
                ],
                vec![10],
            ),
            (
                vec![
                    TestAccount::Real(account1, data1, 0),
                    TestAccount::Real(account2, data2, 0),
                    TestAccount::Real(account3, data3, 0),
                    TestAccount::Duplicate(1),
                ],
                vec![],
            ),
        ];

        for (accounts, instruction_data) in test_cases {
            test_account_parsing(accounts, instruction_data);
        }
    }

    #[test]
    fn test_no_accounts_only_instruction_data() {
        let instruction_data = vec![1, 2, 3, 4];
        let (mut instruction, _program_id) =
            create_test_instruction(vec![], instruction_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Data(data, _) => assert_eq!(data, instruction_data.as_slice()),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_single_account_then_instruction_data() {
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let instruction_data = vec![5, 6, 7, 8];
        let (mut instruction, _program_id) = create_test_instruction(
            vec![TestAccount::Real(account, data, 0)],
            instruction_data.clone(),
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Header(cursor) => {
                let (acc, next_iter) = unsafe { cursor.parse_data() };
                assert_eq!(acc.static_data.is_signer, 1);
                assert_eq!(acc.static_data.is_writable, 1);
                assert_eq!(acc.static_data.data_len, 4);

                match next_iter.next_header() {
                    NextHeader::Data(data, _) => assert_eq!(data, instruction_data.as_slice()),
                    _ => panic!("Expected instruction data"),
                }
            }
            _ => panic!("Expected a real account"),
        }
    }

    #[test]
    fn test_multiple_accounts_with_different_data_lengths() {
        let (account1, data1) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (account2, data2) = create_test_account(false, true, vec![5, 6, 7, 8, 9, 10]);
        let instruction_data = vec![11, 12];
        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1, 0),
                TestAccount::Real(account2, data2, 0),
            ],
            instruction_data.clone(),
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        for _ in 0..2 {
            match iterator.next_header() {
                NextHeader::Header(cursor) => {
                    let (_, next_iter) = unsafe { cursor.parse_data() };
                    iterator = next_iter;
                }
                _ => panic!("Expected a real account"),
            }
        }

        match iterator.next_header() {
            NextHeader::Data(data, _) => assert_eq!(data, instruction_data.as_slice()),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_dup_account() {
        let (account, data) = create_test_account(false, false, vec![1, 2, 3, 4]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Duplicate(0),
                TestAccount::Real(account, data, 0),
            ],
            vec![],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Dup(index, next_iter) => {
                assert_eq!(index, 0);
                iterator = next_iter;
            }
            _ => panic!("Expected a dup account"),
        }

        match iterator.next_header() {
            NextHeader::Header(cursor) => {
                let _ = unsafe { cursor.parse_data() };
            }
            _ => panic!("Expected a real account"),
        }
    }

    #[test]
    fn test_correct_number_of_accounts() {
        let num_accounts = 3;
        let accounts: Vec<TestAccount> = (0..num_accounts)
            .map(|_| {
                TestAccount::Real(
                    create_test_account(false, false, vec![1, 2, 3, 4]).0,
                    vec![1, 2, 3, 4],
                    0,
                )
            })
            .collect();
        let (mut instruction, _program_id) = create_test_instruction(accounts, vec![]);

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let mut count = 0;

        loop {
            match iterator.next_header() {
                NextHeader::Header(cursor) => {
                    count += 1;
                    iterator = unsafe { cursor.skip() };
                }
                NextHeader::Dup(_, next_iter) => {
                    count += 1;
                    iterator = next_iter;
                }
                NextHeader::Data(_, _) => break,
            }
        }

        assert_eq!(count, num_accounts);
    }

    #[test]
    fn test_instruction_length_reading() {
        let instruction_data = vec![1, 2, 3, 4, 5];
        let (mut instruction, _program_id) =
            create_test_instruction(vec![], instruction_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Data(data, _) => assert_eq!(data.len(), instruction_data.len()),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_account_data_offset() {
        let (account1, data1) = create_test_account(false, false, vec![1, 2, 3, 4]);
        let (account2, data2) = create_test_account(false, false, vec![5, 6, 7, 8, 9, 10]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1, 0),
                TestAccount::Real(account2, data2, 0),
            ],
            vec![],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        if let NextHeader::Header(cursor) = iterator.next_header() {
            let (acc1, next_iter) = unsafe { cursor.parse_data() };
            assert_eq!(acc1.static_data.data_len, 4);
            iterator = next_iter;

            if let NextHeader::Header(cursor) = iterator.next_header() {
                let (acc2, _) = unsafe { cursor.parse_data() };
                assert_eq!(acc2.static_data.data_len, 6);
            } else {
                panic!("Expected second account");
            }
        } else {
            panic!("Expected first account");
        }
    }

    #[test]
    fn test_account_data_verification() {
        let (account1, data1) = create_test_account(false, false, vec![1, 2, 3, 4]);
        let (account2, data2) = create_test_account(true, true, vec![5, 6, 7, 8, 9, 10]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1.clone(), 0),
                TestAccount::Real(account2, data2.clone(), 0),
            ],
            vec![],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        if let NextHeader::Header(cursor) = iterator.next_header() {
            let (acc1, next_iter) = unsafe { cursor.parse_data() };
            assert_eq!(acc1.data(), data1);
            iterator = next_iter;

            if let NextHeader::Header(cursor) = iterator.next_header() {
                let (acc2, _) = unsafe { cursor.parse_data() };
                assert_eq!(acc2.data(), data2);
            } else {
                panic!("Expected second account");
            }
        } else {
            panic!("Expected first account");
        }
    }

    #[test]
    fn test_known_next_full_account() {
        let (account, data) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (mut instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data.clone(), 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data(), &[1, 2, 3, 4]);
        assert_eq!(acc.static_data.is_signer, 1);
        assert_eq!(acc.static_data.is_writable, 0);
    }

    #[derive(Copy, Clone)]
    #[repr(C)]
    struct TestStruct {
        a: u32,
        b: u32,
    }

    #[test]
    fn test_like_type_known_next_full_account() {
        let test_data = TestStruct { a: 1, b: 2 };
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                &test_data as *const _ as *const u8,
                std::mem::size_of::<TestStruct>(),
            )
            .to_vec()
        };

        let (account, _) = create_test_account(true, true, data_bytes);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![TestAccount::Real(
                account,
                unsafe {
                    std::slice::from_raw_parts(
                        &test_data as *const _ as *const u8,
                        std::mem::size_of::<TestStruct>(),
                    )
                    .to_vec()
                },
                0,
            )],
            vec![],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<TestStruct>() };

        let data_as_struct = unsafe { &*(acc.data_ptr() as *const TestStruct) };
        assert_eq!(data_as_struct.a, 1);
        assert_eq!(data_as_struct.b, 2);
        assert_eq!(acc.static_data.is_signer, 1);
        assert_eq!(acc.static_data.is_writable, 1);
    }

    // Untyped account tests
    #[test]
    fn test_known_next_full_account_aligned_size() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8]; // size % 8 == 0
        let (account, _) = create_test_account(true, false, data.clone());
        let (mut instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data.clone(), 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.data_len as usize, data.len());
    }

    #[test]
    fn test_known_next_full_account_unaligned_size() {
        let data = vec![1, 2, 3, 4, 5]; // size % 8 != 0
        let (account, _) = create_test_account(true, false, data.clone());
        let (mut instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data.clone(), 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.data_len as usize, data.len());
    }

    #[test]
    fn test_known_next_full_account_zero_size() {
        let data = vec![]; // zero size
        let (account, _) = create_test_account(true, false, data.clone());
        let (mut instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data.clone(), 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.data_len as usize, data.len());
    }

    #[test]
    fn test_like_type_known_next_full_account_aligned() {
        let aligned_data = AlignedStruct { a: 1, b: 2 };
        let mut instruction = create_typed_test_instruction(&aligned_data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<AlignedStruct>() };

        let data_as_struct = unsafe { &*(acc.data_ptr() as *const AlignedStruct) };
        assert_eq!(data_as_struct.a, 1);
        assert_eq!(data_as_struct.b, 2);
        assert_eq!(acc.data().len(), std::mem::size_of::<AlignedStruct>());
    }

    #[test]
    fn test_like_type_known_next_full_account_unaligned() {
        let unaligned_data = UnalignedStruct { a: 1, b: 2 };
        let mut instruction = create_typed_test_instruction(&unaligned_data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<UnalignedStruct>() };

        let data_as_struct = unsafe { &*(acc.data_ptr() as *const UnalignedStruct) };
        assert_eq!(data_as_struct.a, 1);
        assert_eq!(data_as_struct.b, 2);
        assert_eq!(acc.data().len(), std::mem::size_of::<UnalignedStruct>());
    }

    #[test]
    fn test_like_type_known_next_full_account_empty() {
        let empty_data = EmptyStruct {};
        let mut instruction = create_typed_test_instruction(&empty_data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<EmptyStruct>() };

        assert_eq!(acc.data().len(), 0);
        assert_eq!(acc.data().len(), std::mem::size_of::<EmptyStruct>());
    }

    // Tests for all mod 8 size remainders to verify alignment optimization
    #[test]
    fn test_like_type_known_next_full_account_size1() {
        let data = Size1Struct([0x42]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size1Struct>() };

        assert_eq!(acc.data().len(), 1);
        assert_eq!(acc.data()[0], 0x42);
    }

    #[test]
    fn test_like_type_known_next_full_account_size2() {
        let data = Size2Struct([0x12, 0x34]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size2Struct>() };

        assert_eq!(acc.data().len(), 2);
        assert_eq!(acc.data(), &[0x12, 0x34]);
    }

    #[test]
    fn test_like_type_known_next_full_account_size4() {
        let data = Size4Struct([0x11, 0x22, 0x33, 0x44]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size4Struct>() };

        assert_eq!(acc.data().len(), 4);
        assert_eq!(acc.data(), &[0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn test_like_type_known_next_full_account_size5() {
        let data = Size5Struct([0x01, 0x02, 0x03, 0x04, 0x05]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size5Struct>() };

        assert_eq!(acc.data().len(), 5);
        assert_eq!(acc.data(), &[0x01, 0x02, 0x03, 0x04, 0x05]);
    }

    #[test]
    fn test_like_type_known_next_full_account_size6() {
        let data = Size6Struct([0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size6Struct>() };

        assert_eq!(acc.data().len(), 6);
        assert_eq!(acc.data(), &[0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]);
    }

    #[test]
    fn test_like_type_known_next_full_account_size7() {
        let data = Size7Struct([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size7Struct>() };

        assert_eq!(acc.data().len(), 7);
        assert_eq!(acc.data(), &[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70]);
    }

    #[test]
    fn test_like_type_known_next_full_account_size9() {
        let data = Size9Struct([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size9Struct>() };

        assert_eq!(acc.data().len(), 9);
        assert_eq!(
            acc.data(),
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]
        );
    }

    #[test]
    fn test_like_type_known_next_full_account_size15() {
        let data = Size15Struct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let mut instruction = create_typed_test_instruction(&data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.like_type_known_next_full_account::<Size15Struct>() };

        assert_eq!(acc.data().len(), 15);
        assert_eq!(
            acc.data(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    // Test multiple accounts with different sizes to verify pointer advancement
    #[test]
    fn test_like_type_known_next_full_account_mixed_sizes() {
        let data1 = Size1Struct([0xAA]);
        let data7 = Size7Struct([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]);
        let data5 = Size5Struct([0x10, 0x20, 0x30, 0x40, 0x50]);

        let data1_bytes = data1.0.to_vec();
        let data7_bytes = data7.0.to_vec();
        let data5_bytes = data5.0.to_vec();

        let (account1, _) = create_test_account(true, true, data1_bytes.clone());
        let (account7, _) = create_test_account(true, true, data7_bytes.clone());
        let (account5, _) = create_test_account(true, true, data5_bytes.clone());

        let (mut instruction, _) = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1_bytes, 0),
                TestAccount::Real(account7, data7_bytes, 0),
                TestAccount::Real(account5, data5_bytes, 0),
            ],
            vec![0xDE, 0xAD],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        let (acc1, iterator) =
            unsafe { iterator.like_type_known_next_full_account::<Size1Struct>() };
        assert_eq!(acc1.data().len(), 1);
        assert_eq!(acc1.data()[0], 0xAA);

        let (acc7, iterator) =
            unsafe { iterator.like_type_known_next_full_account::<Size7Struct>() };
        assert_eq!(acc7.data().len(), 7);
        assert_eq!(acc7.data(), &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]);

        let (acc5, iterator) =
            unsafe { iterator.like_type_known_next_full_account::<Size5Struct>() };
        assert_eq!(acc5.data().len(), 5);
        assert_eq!(acc5.data(), &[0x10, 0x20, 0x30, 0x40, 0x50]);

        // Verify we can still read instruction data and program address
        let (instr_data, _program_id) =
            unsafe { iterator.known_instruction_data_and_program_address() };
        assert_eq!(instr_data, &[0xDE, 0xAD]);
    }

    #[quickcheck_macros::quickcheck]
    fn quickcheck_mixed_account_types_with_arrays(
        account_types: Vec<TestAccountType>,
        instruction_data_gen: Vec<u8>,
    ) -> bool {
        do_quickcheck_mixed_account_types_with_arrays(account_types, instruction_data_gen)
    }

    #[quickcheck_macros::quickcheck]
    fn quickcheck_compare_with_solana_deserialize(
        account_types: Vec<TestAccountType>,
        instruction_data_gen: Vec<u8>,
    ) -> bool {
        do_quickcheck_compare_with_solana_deserialize(account_types, instruction_data_gen)
    }

    #[test]
    fn test_static_data_helper_methods() {
        let (signer_account, signer_data) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (writable_account, writable_data) = create_test_account(false, true, vec![5, 6, 7, 8]);
        let (both_account, both_data) = create_test_account(true, true, vec![9, 10]);
        let (neither_account, neither_data) = create_test_account(false, false, vec![11, 12]);

        let (mut instruction, _) = create_test_instruction(
            vec![
                TestAccount::Real(signer_account, signer_data, 0),
                TestAccount::Real(writable_account, writable_data, 0),
                TestAccount::Real(both_account, both_data, 0),
                TestAccount::Real(neither_account, neither_data, 0),
            ],
            vec![],
        );

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        let (acc, iter) = unsafe { iter.known_next_full_account() };
        assert!(acc.static_data.is_signer());
        assert!(!acc.static_data.is_writable());
        assert!(!acc.static_data.is_executable());

        let (acc, iter) = unsafe { iter.known_next_full_account() };
        assert!(!acc.static_data.is_signer());
        assert!(acc.static_data.is_writable());

        let (acc, iter) = unsafe { iter.known_next_full_account() };
        assert!(acc.static_data.is_signer());
        assert!(acc.static_data.is_writable());

        let (acc, _) = unsafe { iter.known_next_full_account() };
        assert!(!acc.static_data.is_signer());
        assert!(!acc.static_data.is_writable());
    }

    // Program address tests
    #[test]
    fn test_program_address_retrieval() {
        let instruction_data = vec![1, 2, 3, 4];
        let (mut instruction, expected_program_id) =
            create_test_instruction(vec![], instruction_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Data(data, program_id) => {
                assert_eq!(data, instruction_data.as_slice());
                assert_eq!(*program_id, expected_program_id);
            }
            _ => panic!("Expected instruction data"),
        }
    }

    // ==================== NEW TESTS ====================

    #[test]
    fn test_aligned_known_next_full_account() {
        // Test with 8-byte aligned data (multiple of 8)
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 8 bytes - aligned
        let (account, _) = create_test_account(true, false, data.clone());
        let (mut instruction, _program_id) = create_test_instruction(
            vec![TestAccount::Real(account, data.clone(), 0)],
            vec![9, 10],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, next_iter) = unsafe { iterator.aligned_known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.is_signer, 1);
        assert_eq!(acc.static_data.is_writable, 0);
        assert_eq!(acc.static_data.data_len, 8);

        // Verify we can continue to instruction data
        match next_iter.next_header() {
            NextHeader::Data(instr_data, _) => assert_eq!(instr_data, &[9, 10]),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_aligned_known_next_full_account_multiple() {
        // Test with multiple aligned accounts in sequence
        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 8 bytes
        let data2 = vec![9, 10, 11, 12, 13, 14, 15, 16]; // 8 bytes
        let (account1, _) = create_test_account(true, true, data1.clone());
        let (account2, _) = create_test_account(false, true, data2.clone());
        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1.clone(), 0),
                TestAccount::Real(account2, data2.clone(), 0),
            ],
            vec![],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc1, next_iter) = unsafe { iterator.aligned_known_next_full_account() };
        assert_eq!(acc1.data(), data1.as_slice());

        let (acc2, next_iter) = unsafe { next_iter.aligned_known_next_full_account() };
        assert_eq!(acc2.data(), data2.as_slice());

        match next_iter.next_header() {
            NextHeader::Data(instr_data, _) => assert_eq!(instr_data, &[]),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_known_instruction_data() {
        let instruction_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (account, data) = create_test_account(true, true, vec![10, 20, 30]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![TestAccount::Real(account, data, 0)],
            instruction_data.clone(),
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // Consume the account first
        let (_, next_iter) = unsafe { iterator.known_next_full_account() };

        // Now use known_instruction_data
        let instr_data = unsafe { next_iter.known_instruction_data() };
        assert_eq!(instr_data, instruction_data.as_slice());
    }

    #[test]
    fn test_known_instruction_data_no_accounts() {
        let instruction_data = vec![100, 200, 255, 0, 1];
        let (mut instruction, _program_id) =
            create_test_instruction(vec![], instruction_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // No accounts, so we can call known_instruction_data directly
        let instr_data = unsafe { iterator.known_instruction_data() };
        assert_eq!(instr_data, instruction_data.as_slice());
    }

    #[test]
    fn test_rent_epoch_values() {
        let rent_epoch_value = 12345678901234u64;
        let data = vec![1, 2, 3, 4];
        let (account, _) = create_test_account(true, false, data.clone());
        let mut instruction = create_test_instruction_with_rent_epoch(
            vec![TestAccount::Real(account, data, 0)],
            vec![],
            rent_epoch_value,
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(*acc.rent_epoch, rent_epoch_value);
    }

    #[test]
    fn test_rent_epoch_max_value() {
        let rent_epoch_value = u64::MAX;
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (account, _) = create_test_account(false, true, data.clone());
        let mut instruction = create_test_instruction_with_rent_epoch(
            vec![TestAccount::Real(account, data, 0)],
            vec![],
            rent_epoch_value,
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(*acc.rent_epoch, u64::MAX);
    }

    #[test]
    fn test_remaining_accounts_accessor() {
        let (account1, data1) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (account2, data2) = create_test_account(false, true, vec![5, 6, 7, 8]);
        let (account3, data3) = create_test_account(true, true, vec![9, 10]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1, 0),
                TestAccount::Real(account2, data2, 0),
                TestAccount::Real(account3, data3, 0),
            ],
            vec![],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        assert_eq!(iterator.remaining_accounts(), 3);

        let (_, iter2) = unsafe { iterator.known_next_full_account() };
        assert_eq!(iter2.remaining_accounts(), 2);

        let (_, iter3) = unsafe { iter2.known_next_full_account() };
        assert_eq!(iter3.remaining_accounts(), 1);

        let (_, iter4) = unsafe { iter3.known_next_full_account() };
        assert_eq!(iter4.remaining_accounts(), 0);
    }

    #[test]
    fn test_base_ptr_accessor() {
        let (account, data) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (mut instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![5, 6]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let initial_ptr = iterator.base_ptr();

        // The base_ptr should point somewhere within our instruction buffer
        let buffer_start = instruction.as_ptr();
        let buffer_end = unsafe { buffer_start.add(instruction.len()) };
        assert!(initial_ptr as *const u8 >= buffer_start);
        assert!((initial_ptr as *const u8) < buffer_end);

        // After consuming an account, base_ptr should have advanced
        let (_, next_iter) = unsafe { iterator.known_next_full_account() };
        let next_ptr = next_iter.base_ptr();
        assert!(next_ptr > initial_ptr);
    }

    #[test]
    fn test_max_account_count_256() {
        // Test with exactly 256 accounts (the maximum supported)
        let accounts: Vec<TestAccount> = (0..256)
            .map(|i| {
                let (acc, data) =
                    create_test_account(i % 2 == 0, i % 3 == 0, vec![(i & 0xFF) as u8]);
                TestAccount::Real(acc, data, 0)
            })
            .collect();

        let instruction_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let (mut instruction, _program_id) =
            create_test_instruction(accounts, instruction_data.clone());

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let mut count = 0;

        while iterator.remaining_accounts() > 0 {
            let (acc, next_iter) = unsafe { iterator.known_next_full_account() };
            assert_eq!(acc.data(), &[(count & 0xFF) as u8]);
            iterator = next_iter;
            count += 1;
        }

        assert_eq!(count, 256);

        // Verify instruction data is still accessible
        let instr_data = unsafe { iterator.known_instruction_data() };
        assert_eq!(instr_data, instruction_data.as_slice());
    }

    #[test]
    fn test_multiple_consecutive_dups() {
        // Real account followed by 3 consecutive duplicates
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account, data, 0),
                TestAccount::Duplicate(0),
                TestAccount::Duplicate(0),
                TestAccount::Duplicate(0),
            ],
            vec![99],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // First should be real
        match iterator.next_header() {
            NextHeader::Header(cursor) => {
                let (acc, next) = unsafe { cursor.parse_data() };
                assert_eq!(acc.data(), &[1, 2, 3, 4]);
                iterator = next;
            }
            _ => panic!("Expected real account"),
        }

        // Next three should be dups pointing to index 0
        for _ in 0..3 {
            match iterator.next_header() {
                NextHeader::Dup(idx, next) => {
                    assert_eq!(idx, 0);
                    iterator = next;
                }
                _ => panic!("Expected dup account"),
            }
        }

        // Finally instruction data
        match iterator.next_header() {
            NextHeader::Data(instr_data, _) => assert_eq!(instr_data, &[99]),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_program_address_with_accounts() {
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let instruction_data = vec![5, 6, 7, 8];
        let (mut instruction, expected_program_id) = create_test_instruction(
            vec![TestAccount::Real(account, data, 0)],
            instruction_data.clone(),
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Header(cursor) => {
                let (_, next_iter) = unsafe { cursor.parse_data() };
                match next_iter.next_header() {
                    NextHeader::Data(data, program_id) => {
                        assert_eq!(data, instruction_data.as_slice());
                        assert_eq!(*program_id, expected_program_id);
                    }
                    _ => panic!("Expected instruction data"),
                }
            }
            _ => panic!("Expected a real account"),
        }
    }

    #[test]
    fn test_known_instruction_data_and_program_address() {
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let instruction_data = vec![5, 6, 7, 8];
        let (mut instruction, expected_program_id) = create_test_instruction(
            vec![TestAccount::Real(account, data, 0)],
            instruction_data.clone(),
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (_, iter) = unsafe { iterator.known_next_full_account() };
        let (instr_data, program_id) = unsafe { iter.known_instruction_data_and_program_address() };

        assert_eq!(instr_data, instruction_data.as_slice());
        assert_eq!(*program_id, expected_program_id);
    }

    #[test]
    fn test_known_program_address() {
        let instruction_data = vec![1, 2, 3, 4];
        let (mut instruction, expected_program_id) =
            create_test_instruction(vec![], instruction_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let program_id = unsafe { iterator.known_program_address() };

        assert_eq!(*program_id, expected_program_id);
    }

    #[test]
    fn test_known_program_address_with_accounts() {
        let (account, data) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (mut instruction, expected_program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (_, iter) = unsafe { iterator.known_next_full_account() };
        let program_id = unsafe { iter.known_program_address() };

        assert_eq!(*program_id, expected_program_id);
    }

    #[test]
    fn test_program_address_with_empty_instruction_data() {
        // Edge case: no instruction data, only program address
        let (mut instruction, expected_program_id) = create_test_instruction(vec![], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Data(data, program_id) => {
                assert!(data.is_empty());
                assert_eq!(*program_id, expected_program_id);
            }
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_dups_with_different_indices() {
        // Multiple real accounts, then dups pointing to different indices
        let (account0, data0) = create_test_account(true, false, vec![10]);
        let (account1, data1) = create_test_account(false, true, vec![20]);
        let (account2, data2) = create_test_account(true, true, vec![30]);

        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account0, data0, 0),
                TestAccount::Real(account1, data1, 0),
                TestAccount::Real(account2, data2, 0),
                TestAccount::Duplicate(2), // dup of account2
                TestAccount::Duplicate(0), // dup of account0
                TestAccount::Duplicate(1), // dup of account1
            ],
            vec![],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // Skip past 3 real accounts
        for expected_data in [10u8, 20u8, 30u8] {
            match iterator.next_header() {
                NextHeader::Header(cursor) => {
                    let (acc, next) = unsafe { cursor.parse_data() };
                    assert_eq!(acc.data(), &[expected_data]);
                    iterator = next;
                }
                _ => panic!("Expected real account"),
            }
        }

        // Verify dup indices
        for expected_idx in [2usize, 0usize, 1usize] {
            match iterator.next_header() {
                NextHeader::Dup(idx, next) => {
                    assert_eq!(idx, expected_idx);
                    iterator = next;
                }
                _ => panic!("Expected dup account"),
            }
        }
    }

    #[test]
    fn test_executable_account() {
        let data = vec![0xEF, 0xBE, 0xAD, 0xDE]; // Some "program" data
        let (account, _) = create_test_account_full(false, false, true, 1000000, data.clone());
        let (mut instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data.clone(), 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.static_data.executable, 1);
        assert_eq!(acc.static_data.is_signer, 0);
        assert_eq!(acc.static_data.is_writable, 0);
        assert_eq!(acc.static_data.lamports, 1000000);
        assert_eq!(acc.data(), data.as_slice());
    }

    #[test]
    fn test_lamports_values() {
        let data = vec![1, 2, 3, 4];

        // Test with various lamport values
        for lamports in [0u64, 1, 100, 1_000_000_000, u64::MAX] {
            let (account, _) = create_test_account_full(true, true, false, lamports, data.clone());
            let (mut instruction, _program_id) =
                create_test_instruction(vec![TestAccount::Real(account, data.clone(), 0)], vec![]);

            let iterator =
                unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
            let (acc, _) = unsafe { iterator.known_next_full_account() };

            assert_eq!(acc.static_data.lamports, lamports);
        }
    }

    #[test]
    fn test_program_address_with_large_instruction_data() {
        // Test with larger instruction data payload
        let large_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let (mut instruction, expected_program_id) =
            create_test_instruction(vec![], large_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next_header() {
            NextHeader::Data(data, program_id) => {
                assert_eq!(data.len(), 1024);
                assert_eq!(data, large_data.as_slice());
                assert_eq!(*program_id, expected_program_id);
            }
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_program_address_consistency_between_methods() {
        // Verify known_program_address and known_instruction_data_and_program_address
        // return the same program ID
        let instruction_data = vec![1, 2, 3, 4, 5];
        let (mut instruction1, expected_program_id) =
            create_test_instruction(vec![], instruction_data.clone());
        let mut instruction2 = create_test_instruction_with_program_id(
            vec![],
            instruction_data.clone(),
            expected_program_id,
        );

        let iter1 = unsafe { AccountIterator::new_from_instruction(instruction1.as_mut_ptr()) };
        let iter2 = unsafe { AccountIterator::new_from_instruction(instruction2.as_mut_ptr()) };

        let program_id1 = unsafe { iter1.known_program_address() };
        let (_, program_id2) = unsafe { iter2.known_instruction_data_and_program_address() };

        assert_eq!(*program_id1, *program_id2);
        assert_eq!(*program_id1, expected_program_id);
    }

    #[test]
    fn test_program_address_with_various_instruction_data_lengths() {
        // Test alignment edge cases with different instruction data lengths
        for len in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100] {
            let instruction_data: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
            let (mut instruction, expected_program_id) =
                create_test_instruction(vec![], instruction_data.clone());

            let iterator =
                unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

            match iterator.next_header() {
                NextHeader::Data(data, program_id) => {
                    assert_eq!(data.len(), len);
                    assert_eq!(
                        *program_id, expected_program_id,
                        "Program address mismatch for instruction_data len={}",
                        len
                    );
                }
                _ => panic!("Expected instruction data for len={}", len),
            }
        }
    }

    #[test]
    fn test_mutation_persistence() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (account, _) = create_test_account(true, true, data.clone());
        let (mut instruction, _program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        // Verify original data
        assert_eq!(acc.all_data[0], 1);
        assert_eq!(acc.all_data[3], 4);

        // Mutate the data (all_data is &mut [u8], so we can mutate through it)
        acc.all_data[0] = 100;
        acc.all_data[3] = 200;

        // Re-parse and verify mutation persisted
        let iterator2 = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc2, _) = unsafe { iterator2.known_next_full_account() };

        assert_eq!(acc2.all_data[0], 100);
        assert_eq!(acc2.all_data[3], 200);
    }

    #[test]
    fn test_typed_mutation_persistence() {
        #[derive(Copy, Clone, Pod, Zeroable)]
        #[repr(C)]
        struct MutableData {
            value_a: u64,
            value_b: u64,
        }

        let initial = MutableData {
            value_a: 111,
            value_b: 222,
        };
        let mut instruction = create_typed_test_instruction(&initial);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.static_slurp_typed_account::<MutableData>() };

        // Verify original values
        assert_eq!(acc.data.value_a, 111);
        assert_eq!(acc.data.value_b, 222);

        // Mutate through the typed reference
        acc.data.value_a = 999;
        acc.data.value_b = 888;

        // Re-parse and verify
        let iterator2 = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc2, _) = unsafe { iterator2.static_slurp_typed_account::<MutableData>() };

        assert_eq!(acc2.data.value_a, 999);
        assert_eq!(acc2.data.value_b, 888);
    }

    #[test]
    fn test_large_instruction_data() {
        // Test with 10KB of instruction data
        let instruction_data: Vec<u8> = (0..10240).map(|i| (i & 0xFF) as u8).collect();
        let (account, data) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![TestAccount::Real(account, data, 0)],
            instruction_data.clone(),
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (_, next_iter) = unsafe { iterator.known_next_full_account() };

        let instr_data = unsafe { next_iter.known_instruction_data() };
        assert_eq!(instr_data.len(), 10240);
        assert_eq!(instr_data, instruction_data.as_slice());
    }

    #[test]
    fn test_large_account_data() {
        // Test with account data near MAX_PERMITTED_DATA_INCREASE size
        let large_data: Vec<u8> = (0..1024).map(|i| (i & 0xFF) as u8).collect();
        let (account, _) = create_test_account(true, true, large_data.clone());
        let (mut instruction, _program_id) = create_test_instruction(
            vec![TestAccount::Real(account, large_data.clone(), 0)],
            vec![42],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, next_iter) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data().len(), 1024);
        assert_eq!(acc.data(), large_data.as_slice());
        assert_eq!(acc.static_data.data_len, 1024);

        // Verify we can still access instruction data
        let instr_data = unsafe { next_iter.known_instruction_data() };
        assert_eq!(instr_data, &[42]);
    }

    #[test]
    fn test_static_slurp_typed_account_direct() {
        // Direct unit test for static_slurp_typed_account (not just property test)
        #[derive(Copy, Clone, Pod, Zeroable, PartialEq, Eq, Debug)]
        #[repr(C)]
        struct MyAccount {
            field_a: u64,
            field_b: u64,
            field_c: u64,
        }

        let account_data = MyAccount {
            field_a: 0xDEADBEEF,
            field_b: 0xCAFEBABE,
            field_c: 0x12345678,
        };
        let mut instruction = create_typed_test_instruction(&account_data);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (typed_acc, next_iter) = unsafe { iterator.static_slurp_typed_account::<MyAccount>() };

        assert_eq!(typed_acc.data.field_a, 0xDEADBEEF);
        assert_eq!(typed_acc.data.field_b, 0xCAFEBABE);
        assert_eq!(typed_acc.data.field_c, 0x12345678);
        assert_eq!(
            typed_acc.static_data.data_len,
            std::mem::size_of::<MyAccount>() as u64
        );

        // Should be able to get instruction data after
        assert_eq!(next_iter.remaining_accounts(), 0);
    }

    #[test]
    fn test_static_slurp_typed_accounts_array() {
        // Direct unit test for static_slurp_typed_accounts with array
        #[derive(Copy, Clone, Pod, Zeroable, PartialEq, Eq, Debug)]
        #[repr(C)]
        struct SimpleAccount {
            value: u64,
        }

        // Create 3 accounts with different values
        let values = [100u64, 200u64, 300u64];
        let accounts: Vec<TestAccount> = values
            .iter()
            .map(|&v| {
                let data = SimpleAccount { value: v };
                let data_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &data as *const _ as *const u8,
                        std::mem::size_of::<SimpleAccount>(),
                    )
                    .to_vec()
                };
                let (acc, _) = create_test_account(true, true, data_bytes.clone());
                TestAccount::Real(acc, data_bytes, 0)
            })
            .collect();

        let (mut instruction, _program_id) = create_test_instruction(accounts, vec![42]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (typed_accs, next_iter) =
            unsafe { iterator.static_slurp_typed_accounts::<SimpleAccount, 3>() };

        assert_eq!(typed_accs[0].data.value, 100);
        assert_eq!(typed_accs[1].data.value, 200);
        assert_eq!(typed_accs[2].data.value, 300);

        // Verify remaining state
        assert_eq!(next_iter.remaining_accounts(), 0);
        let instr_data = unsafe { next_iter.known_instruction_data() };
        assert_eq!(instr_data, &[42]);
    }

    #[test]
    fn test_mixed_typed_and_untyped_slurping() {
        // Test real-world pattern: verify authority, then slurp typed accounts
        #[derive(Copy, Clone, Pod, Zeroable)]
        #[repr(C)]
        struct TokenAccount {
            balance: u64,
            owner_index: u64,
        }

        // First account: authority (untyped, just need to check signer)
        let (authority_acc, authority_data) = create_test_account(true, false, vec![0; 32]);

        // Second account: typed token account
        let token_data = TokenAccount {
            balance: 1000000,
            owner_index: 0,
        };
        let token_bytes = unsafe {
            std::slice::from_raw_parts(
                &token_data as *const _ as *const u8,
                std::mem::size_of::<TokenAccount>(),
            )
            .to_vec()
        };
        let (token_acc, _) = create_test_account(false, true, token_bytes.clone());

        let (mut instruction, _program_id) = create_test_instruction(
            vec![
                TestAccount::Real(authority_acc, authority_data, 0),
                TestAccount::Real(token_acc, token_bytes, 0),
            ],
            vec![1, 2, 3], // instruction discriminator
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // Step 1: Get authority account (untyped)
        let (authority, next_iter) = unsafe { iterator.known_next_full_account() };
        assert_eq!(authority.static_data.is_signer, 1);

        // Step 2: Slurp typed token account
        let (token, next_iter) = unsafe { next_iter.static_slurp_typed_account::<TokenAccount>() };
        assert_eq!(token.data.balance, 1000000);
        assert_eq!(token.static_data.is_writable, 1);

        // Step 3: Get instruction data
        let instr_data = unsafe { next_iter.known_instruction_data() };
        assert_eq!(instr_data, &[1, 2, 3]);
    }

    #[test]
    fn test_max_dup_index_254() {
        // Create 255 real accounts, then a dup with index 254
        let mut accounts: Vec<TestAccount> = (0..255)
            .map(|i| {
                let (acc, data) =
                    create_test_account(i % 2 == 0, i % 3 == 0, vec![(i & 0xFF) as u8]);
                TestAccount::Real(acc, data, 0)
            })
            .collect();

        // Add a dup pointing to account 254 (the last real account)
        accounts.push(TestAccount::Duplicate(254));

        let (mut instruction, _program_id) = create_test_instruction(accounts, vec![0xAB]);

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // Skip past the 255 real accounts
        for _ in 0..255 {
            match iterator.next_header() {
                NextHeader::Header(cursor) => {
                    iterator = unsafe { cursor.skip() };
                }
                _ => panic!("Expected real account"),
            }
        }

        // The 256th account should be a dup with index 254
        match iterator.next_header() {
            NextHeader::Dup(idx, next) => {
                assert_eq!(idx, 254);
                iterator = next;
            }
            _ => panic!("Expected dup account with index 254"),
        }

        // Verify instruction data
        match iterator.next_header() {
            NextHeader::Data(data, _) => assert_eq!(data, &[0xAB]),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_program_address_with_multiple_varied_accounts() {
        // Multiple accounts with different data sizes, then verify program address
        let (account1, data1) = create_test_account(true, false, vec![1; 8]); // aligned
        let (account2, data2) = create_test_account(false, true, vec![2; 13]); // unaligned
        let (account3, data3) = create_test_account(true, true, vec![]); // empty
        let (account4, data4) = create_test_account(false, false, vec![4; 100]); // larger

        let instruction_data = vec![0xAB, 0xCD, 0xEF];
        let (mut instruction, expected_program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1, 0),
                TestAccount::Real(account2, data2, 0),
                TestAccount::Real(account3, data3, 0),
                TestAccount::Real(account4, data4, 0),
            ],
            instruction_data.clone(),
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // Skip all 4 accounts
        for _ in 0..4 {
            match iterator.next_header() {
                NextHeader::Header(cursor) => iterator = unsafe { cursor.skip() },
                _ => panic!("Expected an account"),
            }
        }

        match iterator.next_header() {
            NextHeader::Data(data, program_id) => {
                assert_eq!(data, instruction_data.as_slice());
                assert_eq!(*program_id, expected_program_id);
            }
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_very_large_instruction_data() {
        // Test with 64KB of instruction data
        let instruction_data: Vec<u8> = (0..65536).map(|i| (i & 0xFF) as u8).collect();
        let (account, data) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (mut instruction, _program_id) = create_test_instruction(
            vec![TestAccount::Real(account, data, 0)],
            instruction_data.clone(),
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (_, next_iter) = unsafe { iterator.known_next_full_account() };

        let instr_data = unsafe { next_iter.known_instruction_data() };
        assert_eq!(instr_data.len(), 65536);
        assert_eq!(instr_data, instruction_data.as_slice());
    }

    #[test]
    fn test_instruction_data_edge_sizes() {
        // Test instruction data at various edge sizes
        let edge_sizes = [
            0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 255, 256, 1023, 1024, 4095, 4096,
        ];

        for size in edge_sizes {
            let instruction_data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
            let (account, data) = create_test_account(true, true, vec![0xAA]);
            let (mut instruction, _program_id) = create_test_instruction(
                vec![TestAccount::Real(account, data, 0)],
                instruction_data.clone(),
            );

            let iterator =
                unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
            let (_, next_iter) = unsafe { iterator.known_next_full_account() };

            let instr_data = unsafe { next_iter.known_instruction_data() };
            assert_eq!(
                instr_data.len(),
                size,
                "Instruction data size mismatch for size {}",
                size
            );
            assert_eq!(instr_data, instruction_data.as_slice());
        }
    }

    #[test]
    fn test_program_address_with_duplicate_accounts() {
        // Test with duplicate account markers
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let instruction_data = vec![5, 6, 7, 8];
        let (mut instruction, expected_program_id) = create_test_instruction(
            vec![
                TestAccount::Real(account, data, 0),
                TestAccount::Duplicate(0),
                TestAccount::Duplicate(0),
            ],
            instruction_data.clone(),
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // First account is real
        match iterator.next_header() {
            NextHeader::Header(cursor) => {
                iterator = unsafe { cursor.skip() };
            }
            _ => panic!("Expected real account at index 0"),
        }

        // Next two are dups
        for i in 1..3 {
            match iterator.next_header() {
                NextHeader::Dup(idx, next) => {
                    assert_eq!(idx, 0);
                    iterator = next;
                }
                _ => panic!("Expected dup account at index {}", i),
            }
        }

        match iterator.next_header() {
            NextHeader::Data(data, program_id) => {
                assert_eq!(data, instruction_data.as_slice());
                assert_eq!(*program_id, expected_program_id);
            }
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "known_program_address called with")]
    fn test_known_program_address_panics_with_remaining_accounts() {
        // Debug assertion: calling known_program_address before consuming all accounts should panic
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let (mut instruction, _expected_program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // This should panic because there's still 1 account remaining
        let _ = unsafe { iterator.known_program_address() };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "known_instruction_data_and_program_address called with")]
    fn test_known_instruction_data_and_program_address_panics_with_remaining_accounts() {
        // Debug assertion: calling known_instruction_data_and_program_address before consuming all accounts should panic
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let (mut instruction, _expected_program_id) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // This should panic because there's still 1 account remaining
        let _ = unsafe { iterator.known_instruction_data_and_program_address() };
    }

    #[quickcheck_macros::quickcheck]
    fn quickcheck_program_address_always_valid(
        account_types: Vec<TestAccountType>,
        instruction_data_gen: Vec<u8>,
    ) -> bool {
        // Property: program address should always be retrievable and match expected
        let accounts: Vec<_> = account_types
            .iter()
            .flat_map(|t| {
                let accounts = create_account_from_type(t.clone());
                accounts
                    .into_iter()
                    .map(|(acc, data, rent_epoch)| TestAccount::Real(acc, data, rent_epoch))
            })
            .take(256)
            .collect();

        let (mut instruction, expected_program_id) =
            create_test_instruction(accounts, instruction_data_gen.clone());
        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // Iterate through all accounts
        loop {
            match iterator.next_header() {
                NextHeader::Header(cursor) => iterator = unsafe { cursor.skip() },
                NextHeader::Dup(_, next) => iterator = next,
                NextHeader::Data(data, program_id) => {
                    // Verify instruction data
                    if data != instruction_data_gen.as_slice() {
                        return false;
                    }
                    // Verify program address
                    if *program_id != expected_program_id {
                        return false;
                    }
                    return true;
                }
            }
        }
    }

    // ============================================================================
    // AccountHeaderCursor tests
    // ============================================================================

    #[quickcheck_macros::quickcheck]
    fn quickcheck_next_header_parses_correctly(
        account_types: Vec<TestAccountType>,
        instruction_data_gen: Vec<u8>,
    ) -> bool {
        // Property: next_header() should correctly parse all accounts and reach instruction data
        let test_accounts = create_test_accounts_from_types(&account_types);
        if test_accounts.is_empty() {
            return true; // Skip empty case
        }

        let expected_account_count = test_accounts.len();
        let (mut instruction, expected_program_id) =
            create_test_instruction(test_accounts, instruction_data_gen.clone());

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        let mut account_count = 0;

        loop {
            match iterator.next_header() {
                NextHeader::Header(cursor) => {
                    // Verify cursor has valid data length
                    if cursor.data_len() > 10 * 1024 * 1024 {
                        return false; // Unreasonably large
                    }
                    let (_, next) = unsafe { cursor.parse_data() };
                    iterator = next;
                    account_count += 1;
                }
                NextHeader::Dup(idx, next) => {
                    // Dup index should be less than current account count
                    if idx >= account_count {
                        return false;
                    }
                    iterator = next;
                    account_count += 1;
                }
                NextHeader::Data(data, program_id) => {
                    // Verify we parsed the expected number of accounts
                    if account_count != expected_account_count {
                        return false;
                    }
                    // Verify instruction data matches
                    if data != instruction_data_gen.as_slice() {
                        return false;
                    }
                    // Verify program id matches
                    if *program_id != expected_program_id {
                        return false;
                    }
                    return true;
                }
            }
        }
    }

    #[test]
    fn test_cursor_size_inspection() {
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let (mut instruction, _) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let cursor = unsafe { iter.known_next_header() };

        assert_eq!(cursor.data_len(), 8);
        assert!(cursor.has_min_size(1));
        assert!(cursor.has_min_size(8));
        assert!(!cursor.has_min_size(9));
        assert!(cursor.has_exact_size(8));
        assert!(!cursor.has_exact_size(7));
        assert!(cursor.has_size_of::<u64>());
        assert!(!cursor.has_size_of::<u32>());
    }

    #[test]
    fn test_cursor_peek_bytes() {
        let (account, data) = create_test_account(true, true, vec![0xAA, 0xBB, 0xCC, 0xDD]);
        let (mut instruction, _) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let cursor = unsafe { iter.known_next_header() };

        // Peek at first 2 bytes
        let peeked = cursor.peek_bytes(2).unwrap();
        assert_eq!(peeked, &[0xAA, 0xBB]);

        // Peek at all 4 bytes
        let peeked = cursor.peek_bytes(4).unwrap();
        assert_eq!(peeked, &[0xAA, 0xBB, 0xCC, 0xDD]);

        // Peek beyond bounds returns None
        assert!(cursor.peek_bytes(5).is_none());

        // Can still parse after peeking
        let (acc, _) = unsafe { cursor.parse_data() };
        assert_eq!(acc.data(), &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_cursor_peek_as() {
        let data_bytes: Vec<u8> = 0x12345678u64.to_le_bytes().to_vec();
        let (account, data) = create_test_account(true, true, data_bytes);
        let (mut instruction, _) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let cursor = unsafe { iter.known_next_header() };

        // peek_as with correct size
        let peeked: &u64 = cursor.peek_as().unwrap();
        assert_eq!(*peeked, 0x12345678u64);

        // peek_as with wrong size returns None
        let wrong: Option<&u32> = cursor.peek_as();
        assert!(wrong.is_none());
    }

    #[test]
    fn test_cursor_skip() {
        let (acc1, data1) = create_test_account(true, false, vec![1, 2, 3, 4]);
        let (acc2, data2) = create_test_account(false, true, vec![5, 6, 7, 8]);
        let (mut instruction, _) = create_test_instruction(
            vec![
                TestAccount::Real(acc1, data1, 0),
                TestAccount::Real(acc2, data2, 0),
            ],
            vec![99],
        );

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // Skip first account via cursor
        let cursor = unsafe { iter.known_next_header() };
        assert!(cursor.static_data.is_signer());
        let iter = unsafe { cursor.skip() };

        // Second account should be accessible
        let cursor = unsafe { iter.known_next_header() };
        assert!(cursor.static_data.is_writable());
        let (acc, iter) = unsafe { cursor.parse_data() };
        assert_eq!(acc.data(), &[5, 6, 7, 8]);

        // Should reach instruction data
        match iter.next_header() {
            NextHeader::Data(data, _) => assert_eq!(data, &[99]),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_cursor_parse_typed_checked() {
        let data_bytes: Vec<u8> = QuickCheckAligned { a: 42, b: 99 }.to_vec();
        let (account, data) = create_test_account(true, true, data_bytes);
        let (mut instruction, _) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let cursor = unsafe { iter.known_next_header() };

        // Correct type should succeed
        match unsafe { cursor.parse_typed_checked::<QuickCheckAligned>() } {
            Ok((typed, _)) => {
                assert_eq!(typed.data.a, 42);
                assert_eq!(typed.data.b, 99);
            }
            Err(_) => panic!("Expected successful parse"),
        }
    }

    #[test]
    fn test_cursor_parse_typed_checked_wrong_size() {
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]); // 4 bytes
        let (mut instruction, _) =
            create_test_instruction(vec![TestAccount::Real(account, data, 0)], vec![]);

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let cursor = unsafe { iter.known_next_header() };

        // Wrong size should return Err with cursor
        match unsafe { cursor.parse_typed_checked::<QuickCheckAligned>() } {
            Ok(_) => panic!("Expected error for wrong size"),
            Err(returned_cursor) => {
                // Can still use the returned cursor
                assert_eq!(returned_cursor.data_len(), 4);
                let iter = unsafe { returned_cursor.skip() };
                assert_eq!(iter.remaining_accounts(), 0);
            }
        }
    }

    #[test]
    fn test_next_header_with_dups() {
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let (mut instruction, _) = create_test_instruction(
            vec![
                TestAccount::Real(account, data, 0),
                TestAccount::Duplicate(0),
                TestAccount::Duplicate(0),
            ],
            vec![],
        );

        let iter = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // First: real account
        match iter.next_header() {
            NextHeader::Header(cursor) => {
                assert!(cursor.static_data.is_signer());
                let (_, iter) = unsafe { cursor.parse_data() };

                // Second: dup
                match iter.next_header() {
                    NextHeader::Dup(idx, iter) => {
                        assert_eq!(idx, 0);

                        // Third: dup
                        match iter.next_header() {
                            NextHeader::Dup(idx, iter) => {
                                assert_eq!(idx, 0);

                                // Finally: data
                                match iter.next_header() {
                                    NextHeader::Data(_, _) => {}
                                    _ => panic!("Expected data"),
                                }
                            }
                            _ => panic!("Expected dup"),
                        }
                    }
                    _ => panic!("Expected dup"),
                }
            }
            _ => panic!("Expected header"),
        }
    }
}
