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

use std::marker::PhantomData;

use crate::solana_export::constants::MAX_PERMITTED_DATA_INCREASE;
use crate::solana_export::constants::{BPF_ALIGN_OF_U128, NON_DUP_MARKER};
use crate::solana_export::pubkey::Pubkey;
use bytemuck::{Pod, Zeroable};

use crate::{assume, bytes::slurp};

/// Represents an account encountered during iteration.
/// It can either be a full account descriptor or a reference to a previously seen duplicate.
pub enum AccountInInstruction {
    /// Contains the full metadata and data pointer for a non-duplicate account.
    RealAccount(NonDupAccount<'static>),
    /// Indicates a duplicate account, referencing the index of the first occurrence.
    Dup(usize),
}

/// Represents the result of advancing the `AccountIterator`.
/// It can be either the next account in the sequence or the final instruction data.
pub enum NextAccount {
    /// The instruction data slice, returned after all accounts have been processed.
    Data(&'static mut [u8]),
    /// The next account encountered and the iterator advanced past it.
    Account(AccountInInstruction, AccountIterator),
}

/// An iterator over the serialized account data passed into a Solana program entrypoint.
///
/// This iterator provides `unsafe` methods for efficiently accessing account metadata
/// and data directly from the raw byte buffer provided by the Solana runtime.
/// It handles parsing duplicate account markers and distinguishing between account
/// data and the final instruction data segment.
///
/// Use `AccountIterator::new_from_instruction` to create an iterator from the
/// beginning of the runtime-provided buffer.
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
        let known_size = std::mem::size_of::<T>() as u64;
        unsafe {
            assume!(known_size == *size_in_data, "Known size is not real size");
        }
        known_size as u64
    }

    #[inline]
    fn get_next_pointer(ptr: *mut u8) -> *mut u8 {
        if std::mem::size_of::<T>() % BPF_ALIGN_OF_U128 == 0 || std::mem::size_of::<T>() == 0 {
            ptr
        } else {
            unsafe { ptr.add(ptr.align_offset(BPF_ALIGN_OF_U128)) }
        }
    }
}

impl AccountIterator {
    /// Creates a new AccountIterator from the instruction data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer is valid. It's only safe to call this on data
    /// generated by the Solana runtime.
    ///
    /// # Arguments
    ///
    /// * `base` - A pointer to the start of the instruction data buffer provided by the runtime.
    ///
    /// # Returns
    ///
    /// A new AccountIterator instance positioned at the first account descriptor.
    #[inline]
    pub unsafe fn new_from_instruction(base: *mut u8) -> Self {
        let (num_accounts, next_ptr) = unsafe { slurp::<u64>(base) };
        Self::new_from_raw(next_ptr, *num_accounts as usize)
    }

    /// Creates a new AccountIterator from raw pointer and account count.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it works with raw pointers.
    /// The caller must ensure that `base_ptr` points to valid, properly formatted
    /// account data (or instruction data if `remaining_accounts` is 0) and that
    /// `remaining_accounts` accurately reflects the number of accounts following `base_ptr`.
    ///
    /// # Arguments
    ///
    /// * `base_ptr` - A pointer to the start of the account data.
    /// * `remaining_accounts` - The number of remaining accounts to iterate over.
    ///
    /// # Returns
    ///
    /// A new AccountIterator instance.
    #[inline]
    pub unsafe fn new_from_raw(base_ptr: *mut u8, remaining_accounts: usize) -> Self {
        Self {
            base_ptr,
            remaining_accounts,
        }
    }

    #[inline]
    /// Clones the iterator
    ///
    /// # Safety
    ///
    /// This function is unsafe as it exposes calls that give one mutable access to the underlying bytes
    /// Ensure that you don't give yourself multiple mutable references
    pub unsafe fn unsafe_clone(&self) -> Self {
        Self {
            base_ptr: self.base_ptr,
            remaining_accounts: self.remaining_accounts,
        }
    }

    /// Advances the iterator and returns the next account or instruction data.
    ///
    /// # Returns
    ///
    /// A NextAccount enum containing either the next account or the instruction data.
    #[inline(always)]
    pub fn next(self) -> NextAccount {
        if self.remaining_accounts == 0 {
            let slice = self.slurp_instruction_data();
            NextAccount::Data(slice)
        } else {
            let is_dup = unsafe { *self.base_ptr };
            let (acc, next) = if is_dup == NON_DUP_MARKER {
                let (non_dup, next) = unsafe { self.slurp_real_account::<Dynamic>() };
                (AccountInInstruction::RealAccount(non_dup), next)
            } else {
                let next = unsafe { self.base_ptr.add(8) };
                (AccountInInstruction::Dup(is_dup as usize), next)
            };
            NextAccount::Account(acc, unsafe {
                AccountIterator::new_from_raw(next, self.remaining_accounts - 1)
            })
        }
    }

    /// Retrieves the next full (non-duplicate) account.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes the next account is a full account.
    /// The caller must ensure that this assumption holds true.
    ///
    /// # Returns
    ///
    /// A tuple containing the next full account and the updated iterator.
    #[inline]
    pub unsafe fn known_next_full_account(self) -> (NonDupAccount<'static>, AccountIterator) {
        debug_assert!(self.remaining_accounts > 0);

        let is_dup = unsafe { &*self.base_ptr };

        debug_assert_eq!(*is_dup, NON_DUP_MARKER);

        let (account, next) = self.slurp_real_account::<Dynamic>();

        (account, unsafe {
            AccountIterator::new_from_raw(next, self.remaining_accounts - 1)
        })
    }

    /// Retrieves the next full (non-duplicate) account.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes the next account is a full account.
    /// This also assumes that the account is aligned to an 8 byte boundary
    /// The caller must ensure that this assumption holds true.
    ///
    /// # Returns
    ///
    /// A tuple containing the next full account and the updated iterator.
    #[inline]
    pub unsafe fn aligned_known_next_full_account(
        self,
    ) -> (NonDupAccount<'static>, AccountIterator) {
        debug_assert!(self.remaining_accounts > 0);

        let is_dup = unsafe { &*self.base_ptr };

        debug_assert_eq!(*is_dup, NON_DUP_MARKER);

        let (account, next) = self.slurp_real_account::<Aligned>();

        // Verify alignment assumption - compiler can use this hint to eliminate alignment code
        assume!(
            (next as usize) % BPF_ALIGN_OF_U128 == 0,
            "aligned_known_next_full_account: next pointer not 8-byte aligned"
        );

        (account, unsafe {
            AccountIterator::new_from_raw(next, self.remaining_accounts - 1)
        })
    }

    /// Retrieves the next full (non-duplicate) account. This call assumes that the account
    /// exactly holds one of the type passed with no extra allocated data
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes the next account is a full account,
    /// and that the account exactly holds one of the type passed with no extra allocated data
    ///
    /// The caller must ensure that this assumption holds true.
    ///
    /// # Returns
    ///
    /// A tuple containing the next full account and the updated iterator.
    #[inline]
    pub unsafe fn typed_known_next_full_account<T>(
        self,
    ) -> (NonDupAccount<'static>, AccountIterator) {
        debug_assert!(self.remaining_accounts > 0);

        let is_dup = unsafe { &*self.base_ptr };

        debug_assert_eq!(*is_dup, NON_DUP_MARKER);

        let (account, next) = self.slurp_real_account::<TypedSlurper<T>>();

        (
            account,
            AccountIterator::new_from_raw(next, self.remaining_accounts - 1),
        )
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

    /// Retrieves the instruction data, skipping the normal iteration.
    ///
    /// Use this when you've already processed all accounts and want direct access to
    /// the instruction data without going through `NextAccount::Data`.
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
    /// A mutable slice containing the instruction data.
    #[inline]
    pub unsafe fn known_instruction_data(self) -> &'static mut [u8] {
        debug_assert_eq!(
            self.remaining_accounts, 0,
            "known_instruction_data called with {} accounts remaining",
            self.remaining_accounts
        );
        self.slurp_instruction_data()
    }

    /// Helper function to retrieve a real (non-duplicate) account.
    ///
    /// # Returns
    ///
    /// A tuple containing the NonDupAccount and a pointer to the next data.
    #[inline]
    unsafe fn slurp_real_account<S: Slurper>(&self) -> (NonDupAccount<'static>, *mut u8) {
        let (account_static, next) = unsafe { slurp::<NonDupAccountStatic>(self.base_ptr) };
        let data_len = S::get_account_size(&account_static.data_len);
        let data = std::slice::from_raw_parts_mut(next, data_len as usize);

        let next = next.add(account_static.data_len as usize + MAX_PERMITTED_DATA_INCREASE);

        let next = S::get_next_pointer(next);

        let (rent_epoch, next) = slurp::<u64>(next);

        let account = NonDupAccount {
            static_data: account_static,
            all_data: data,
            rent_epoch,
        };

        (account, next)
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
            std::mem::size_of::<T>() % 8,
            0,
            "Account size must be a multiple of 8"
        );
        assert_eq!(
            std::mem::size_of::<TypedNonDupAccount<T>>() % 8,
            0,
            "Account array size must be a multiple of 8"
        );
        assert_eq!(
            std::mem::size_of::<[TypedNonDupAccount<T>; N]>() % 8,
            0,
            "Account array size must be a multiple of 8"
        );

        let (data, next_bytes) = slurp::<[TypedNonDupAccount<T>; N]>(self.base_ptr);

        #[cfg(debug_assertions)]
        {
            let mut copy_of_self = self.unsafe_clone();

            for _ in 0..N {
                match copy_of_self.next() {
                    NextAccount::Account(acc, next_iter) => {
                        copy_of_self = next_iter;
                        match acc {
                            AccountInInstruction::RealAccount(acc) => {
                                if acc.static_data.data_len != std::mem::size_of::<T>() as u64 {
                                    panic!(
                                        "Expected account size of {} but got {}",
                                        std::mem::size_of::<T>() as u64,
                                        acc.static_data.data_len
                                    );
                                }
                            }
                            AccountInInstruction::Dup(_) => {
                                panic!("Expected a real account, found a duplicate")
                            }
                        }
                    }
                    NextAccount::Data(_) => panic!("Expected an account, found instruction data"),
                }
            }

            assert_eq!(next_bytes, copy_of_self.base_ptr);
        }

        (data, next_bytes)
    }

    /// Helper function to retrieve the instruction data.
    ///
    /// # Returns
    ///
    /// A slice containing the instruction data.
    #[inline]
    fn slurp_instruction_data(self) -> &'static mut [u8] {
        let (instruction_length, instruction_data) = unsafe { slurp::<u64>(self.base_ptr) };
        unsafe { std::slice::from_raw_parts_mut(instruction_data, *instruction_length as usize) }
    }

    /// Returns the number of remaining accounts to be processed.
    ///
    /// # Returns
    ///
    /// The count of remaining accounts.
    #[inline]
    pub fn remaining_accounts(&self) -> usize {
        self.remaining_accounts
    }

    /// Returns the current base pointer of the iterator.
    ///
    /// # Returns
    ///
    /// A pointer to the current position in the data.
    #[inline]
    pub fn base_ptr(&self) -> *mut u8 {
        self.base_ptr
    }
}

/// Represents the static part of a non-duplicate account's metadata in the
/// serialized buffer format.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Eq, Debug)]
pub struct NonDupAccountStatic {
    /// Distinguishes between duplicate (index) and non-duplicate (`NON_DUP_MARKER`) entries.
    pub is_dup: u8,
    /// Non-zero if the account signed the transaction.
    pub is_signer: u8,
    /// Non-zero if the account is writable.
    pub is_writable: u8,
    /// Non-zero if the account holds an executable program.
    pub executable: u8,
    /// Padding to ensure 8-byte alignment.
    pub original_data_len: u32,
    /// The public key of the account.
    pub key: Pubkey,
    /// The public key of the account's owner program.
    pub owner: Pubkey,
    /// The number of lamports held by the account.
    pub lamports: u64,
    /// The length of the account's data slice.
    pub data_len: u64,
}

/// A reference to a non-duplicate account's data parsed from the instruction buffer.
#[derive(PartialEq, Eq)]
pub struct NonDupAccount<'a> {
    /// Reference to the static metadata part of the account.
    pub static_data: &'a NonDupAccountStatic,
    /// Mutable slice containing the account's data. Note that this slice's capacity
    /// includes the `MAX_PERMITTED_DATA_INCREASE` buffer.
    pub all_data: &'a mut [u8],
    /// Reference to the rent epoch field associated with the account.
    pub rent_epoch: &'a u64,
}

/// Typed account for direct memory mapping. Layout matches the runtime's serialization exactly.
///
/// ```text
/// [NonDupAccountStatic: 128 bytes]
/// [data: T]
/// [_buffer: 10KB growth reserve]
/// [rent_epoch: u64]
/// ```
///
/// The `_buffer` exists because Solana reserves 10KB after each account for potential
/// reallocation during execution.
#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(C)]
pub struct TypedNonDupAccount<T: Pod + Zeroable> {
    /// The static metadata: is_signer, is_writable, key, owner, lamports, data_len, etc.
    pub static_data: NonDupAccountStatic,
    /// The account's data, interpreted as type `T`. Mutations here write directly to the
    /// runtime buffer and persist after the instruction completes.
    pub data: T,
    /// Reserved buffer space for potential data growth. Do not access directly.
    _buffer: [u8; MAX_PERMITTED_DATA_INCREASE],
    /// The rent epoch associated with the account.
    pub rent_epoch: u64,
}

unsafe impl<T: Pod + Zeroable> Pod for TypedNonDupAccount<T> {}
unsafe impl<T: Pod + Zeroable> Zeroable for TypedNonDupAccount<T> {}

impl NonDupAccount<'_> {
    /// Returns an immutable slice of the account's data.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.all_data
    }

    /// Returns a const pointer to the beginning of the account's data.
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.all_data.as_ptr()
    }

    /// Returns a mutable pointer to the beginning of the account's data.
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.all_data.as_mut_ptr()
    }
}

#[cfg(any(test, fuzzing))]
pub mod arbitrary_impls {

    #[cfg(all(test, fuzzing))]
    compile_error!("fuzzing and test cannot both be true");

    use crate::solana_export::{self, pubkey_from_array, unique_pubkey, IsAccount};

    use crate::bytes::PodUtils;

    use super::*;
    #[cfg(test)]
    use quickcheck::Arbitrary;

    #[derive(Clone, Debug)]
    pub enum TestAccount {
        Real(NonDupAccountStatic, Vec<u8>),
        Duplicate(u8),
    }

    #[cfg(test)]
    fn arbitrary_array<T: Arbitrary>(g: &mut quickcheck::Gen) -> [T; 32] {
        std::array::from_fn(|_| T::arbitrary(g))
    }

    #[cfg(test)]
    impl Arbitrary for TestAccount {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let should_be_dup = u8::arbitrary(g) < 200;
            if should_be_dup {
                let index = u8::arbitrary(g);
                let index = index.wrapping_add((index == NON_DUP_MARKER) as u8);
                TestAccount::Duplicate(index)
            } else {
                let data = Vec::<u8>::arbitrary(g);
                let static_data = NonDupAccountStatic {
                    is_dup: NON_DUP_MARKER,
                    is_signer: bool::arbitrary(g) as u8,
                    is_writable: bool::arbitrary(g) as u8,
                    executable: 0,
                    original_data_len: data.len() as u32,
                    key: pubkey_from_array(arbitrary_array(g)),
                    owner: pubkey_from_array(arbitrary_array(g)),
                    lamports: 100,
                    data_len: data.len() as u64,
                };
                TestAccount::Real(static_data, data)
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
            is_dup: NON_DUP_MARKER,
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

    pub fn create_test_instruction(
        accounts: Vec<TestAccount>,
        instruction_data: Vec<u8>,
    ) -> Vec<u8> {
        create_test_instruction_with_rent_epoch(accounts, instruction_data, 0)
    }

    pub fn create_test_instruction_with_rent_epoch(
        accounts: Vec<TestAccount>,
        instruction_data: Vec<u8>,
        rent_epoch: u64,
    ) -> Vec<u8> {
        let mut instruction = Vec::new();
        let num_accounts = accounts.len() as u64;
        instruction.extend_from_slice(&num_accounts.to_le_bytes());

        for account in accounts {
            match account {
                TestAccount::Real(acc, data) => {
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
        // add a program ID and some padding for alignment
        instruction.extend_from_slice(&[0; 64]);
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

    pub fn create_typed_test_instruction<T: Copy>(data: &T) -> Vec<u8> {
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data as *const _ as *const u8, std::mem::size_of::<T>())
                .to_vec()
        };

        let (account, _) = create_test_account(true, true, data_bytes.clone());
        create_test_instruction(vec![TestAccount::Real(account, data_bytes)], vec![])
    }
    #[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(C)]
    pub struct QuickCheckAligned {
        pub a: u64,
        pub b: u64,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
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
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    #[repr(C)]
    pub struct QuickCheckEmpty {}

    #[derive(Debug, Clone)]
    #[cfg_attr(fuzzing, derive(arbitrary::Arbitrary))]
    pub enum TestAccountType {
        Aligned(QuickCheckAligned, bool, bool),
        Unaligned(QuickCheckUnaligned, bool, bool),
        Empty(QuickCheckEmpty, bool, bool),
        Untyped(Vec<u8>, bool, bool),
        AlignedArray([(QuickCheckAligned, bool, bool); 2]),
        AlignedArray3([(QuickCheckAligned, bool, bool); 3]),
        /// Duplicate of a previous account. The u8 is a hint that will be
        /// taken modulo the number of real accounts seen so far.
        Duplicate(u8),
    }

    #[cfg(test)]
    impl Arbitrary for TestAccountType {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            fn aligned(g: &mut quickcheck::Gen) -> (QuickCheckAligned, bool, bool) {
                (
                    QuickCheckAligned {
                        a: u64::arbitrary(g),
                        b: u64::arbitrary(g),
                    },
                    bool::arbitrary(g),
                    bool::arbitrary(g),
                )
            }
            match u8::arbitrary(g) % 8 {
                0 => TestAccountType::Aligned(
                    QuickCheckAligned {
                        a: u64::arbitrary(g),
                        b: u64::arbitrary(g),
                    },
                    bool::arbitrary(g),
                    bool::arbitrary(g),
                ),
                1 => TestAccountType::Unaligned(
                    QuickCheckUnaligned {
                        a: u16::arbitrary(g),
                        b: u8::arbitrary(g),
                    },
                    bool::arbitrary(g),
                    bool::arbitrary(g),
                ),
                2 => TestAccountType::Empty(
                    QuickCheckEmpty {},
                    bool::arbitrary(g),
                    bool::arbitrary(g),
                ),
                3 => {
                    let len = usize::arbitrary(g) % 32;
                    let data: Vec<u8> = (0..len).map(|_| u8::arbitrary(g)).collect();
                    TestAccountType::Untyped(data, bool::arbitrary(g), bool::arbitrary(g))
                }
                4 => TestAccountType::AlignedArray([aligned(g), aligned(g)]),
                5 => TestAccountType::AlignedArray3([aligned(g), aligned(g), aligned(g)]),
                // ~25% chance of generating a duplicate (2 out of 8 cases)
                _ => TestAccountType::Duplicate(u8::arbitrary(g)),
            }
        }
    }

    pub fn create_account_from_type(
        account_type: TestAccountType,
    ) -> Vec<(NonDupAccountStatic, Vec<u8>)> {
        match account_type {
            TestAccountType::Aligned(aligned, is_signer, is_writable) => {
                let data = aligned.to_vec();
                vec![create_test_account(is_signer, is_writable, data)]
            }
            TestAccountType::Unaligned(unaligned, is_signer, is_writable) => {
                let data = unaligned.create_vec();
                vec![create_test_account(is_signer, is_writable, data)]
            }
            TestAccountType::Empty(_, is_signer, is_writable) => {
                vec![create_test_account(is_signer, is_writable, vec![])]
            }
            TestAccountType::Untyped(data, is_signer, is_writable) => {
                vec![create_test_account(is_signer, is_writable, data.clone())]
            }
            TestAccountType::AlignedArray(array) => array
                .iter()
                .map(|(aligned, is_signer, is_writable)| {
                    let data = aligned.to_vec();
                    create_test_account(*is_signer, *is_writable, data)
                })
                .collect(),
            TestAccountType::AlignedArray3(array) => array
                .iter()
                .map(|(aligned, is_signer, is_writable)| {
                    let data = aligned.to_vec();
                    create_test_account(*is_signer, *is_writable, data)
                })
                .collect(),
            // Duplicates are handled separately in create_test_accounts_from_types
            TestAccountType::Duplicate(_) => vec![],
        }
    }

    /// Converts a list of TestAccountType into a flat list of TestAccount,
    /// properly handling duplicates by computing valid indices based on
    /// how many real accounts have been seen so far.
    pub fn create_test_accounts_from_types(account_types: &[TestAccountType]) -> Vec<TestAccount> {
        let mut result = Vec::new();
        let mut real_account_count = 0usize;

        for account_type in account_types {
            match account_type {
                TestAccountType::Duplicate(hint) => {
                    // Only create a dup if there's at least one real account to reference
                    if real_account_count > 0 {
                        let dup_index = (*hint as usize) % real_account_count;
                        result.push(TestAccount::Duplicate(dup_index as u8));
                    }
                    // If no real accounts yet, skip this dup (can't reference anything)
                }
                other => {
                    let real_accounts = create_account_from_type(other.clone());
                    for (acc, data) in real_accounts {
                        result.push(TestAccount::Real(acc, data));
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
                    .map(|(acc, data)| TestAccount::Real(acc, data))
            })
            .collect();

        let mut instruction = create_test_instruction(accounts, instruction_data_gen.clone());
        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        for account_type in account_types {
            match account_type {
                TestAccountType::Aligned(al, is_signer, is_writable) => {
                    let (acc, next) =
                        unsafe { iterator.static_slurp_typed_account::<QuickCheckAligned>() };
                    if acc.static_data.data_len != std::mem::size_of::<QuickCheckAligned>() as u64 {
                        return false;
                    }
                    if (acc.static_data.is_signer == 1) != is_signer {
                        return false;
                    }
                    if (acc.static_data.is_writable == 1) != is_writable {
                        return false;
                    }
                    if acc.data != al {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::Unaligned(un, is_signer, is_writable) => {
                    let (acc, next) =
                        unsafe { iterator.typed_known_next_full_account::<QuickCheckUnaligned>() };
                    if acc.static_data.data_len != std::mem::size_of::<QuickCheckUnaligned>() as u64
                    {
                        return false;
                    }
                    if (acc.static_data.is_signer == 1) != is_signer {
                        return false;
                    }
                    if (acc.static_data.is_writable == 1) != is_writable {
                        return false;
                    }
                    let data_as_struct =
                        unsafe { &*(acc.data().as_ptr() as *const QuickCheckUnaligned) };
                    if data_as_struct != &un {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::Empty(_, is_signer, is_writable) => {
                    let (acc, next) =
                        unsafe { iterator.typed_known_next_full_account::<QuickCheckEmpty>() };
                    if acc.static_data.data_len != 0 {
                        return false;
                    }
                    if (acc.static_data.is_signer == 1) != is_signer {
                        return false;
                    }
                    if (acc.static_data.is_writable == 1) != is_writable {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::Untyped(data, is_signer, is_writable) => {
                    let (acc, next) = unsafe { iterator.known_next_full_account() };
                    if acc.data() != data.as_slice() {
                        return false;
                    }
                    if (acc.static_data.is_signer == 1) != is_signer {
                        return false;
                    }
                    if (acc.static_data.is_writable == 1) != is_writable {
                        return false;
                    }
                    iterator = next;
                }
                TestAccountType::AlignedArray(array) => {
                    let (accs, next) =
                        unsafe { iterator.static_slurp_typed_accounts::<QuickCheckAligned, 2>() };
                    for ((given, is_signer, is_writable), acc) in array.iter().zip(accs.iter()) {
                        if acc.static_data.data_len
                            != std::mem::size_of::<QuickCheckAligned>() as u64
                        {
                            return false;
                        }
                        if (acc.static_data.is_signer == 1) != *is_signer {
                            return false;
                        }
                        if (acc.static_data.is_writable == 1) != *is_writable {
                            return false;
                        }
                        if &acc.data != given {
                            return false;
                        }
                    }
                    iterator = next;
                }
                TestAccountType::AlignedArray3(array) => {
                    let (accs, next) =
                        unsafe { iterator.static_slurp_typed_accounts::<QuickCheckAligned, 3>() };
                    for ((given, is_signer, is_writable), acc) in array.iter().zip(accs.iter()) {
                        if acc.static_data.data_len
                            != std::mem::size_of::<QuickCheckAligned>() as u64
                        {
                            return false;
                        }
                        if (acc.static_data.is_signer == 1) != *is_signer {
                            return false;
                        }
                        if (acc.static_data.is_writable == 1) != *is_writable {
                            return false;
                        }
                        if &acc.data != given {
                            return false;
                        }
                    }
                    iterator = next;
                }
                TestAccountType::Duplicate(_) => {
                    // Duplicates are filtered out before this loop, so this is unreachable
                    unreachable!("Duplicate variants should have been filtered out")
                }
            }
        }

        // Verify we've reached the instruction data
        let NextAccount::Data(check) = iterator.next() else {
            return false;
        };
        check == instruction_data_gen.as_slice()
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
        let mut instruction_buffer =
            create_test_instruction(test_accounts.clone(), instruction_data_gen.clone());

        // 3. Parse with AccountIterator
        let mut fast_results = Vec::new();
        let mut fast_iter =
            unsafe { AccountIterator::new_from_instruction(instruction_buffer.as_mut_ptr()) };
        let final_fast_data = loop {
            match fast_iter.next() {
                NextAccount::Account(acc, next_iter) => {
                    fast_results.push(acc);
                    fast_iter = next_iter;
                }
                NextAccount::Data(data) => {
                    break data.to_vec(); // Clone data for comparison
                }
            }
        };

        // 4. Parse with solana_program::entrypoint::deserialize
        let (_program_id_solana, accounts_solana, instruction_data_solana) =
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
                AccountInInstruction::RealAccount(fast_real) => {
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
                AccountInInstruction::Dup(dup_index) => {
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

        true // All checks passed
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use super::arbitrary_impls::*;

    fn test_account_parsing(accounts: Vec<TestAccount>, instruction_data: Vec<u8>) {
        let mut instruction = create_test_instruction(accounts.clone(), instruction_data.clone());
        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        for expected_account in accounts {
            match iterator.next() {
                NextAccount::Account(parsed_account, next_iter) => {
                    match (expected_account, parsed_account) {
                        (
                            TestAccount::Real(expected, expected_data),
                            AccountInInstruction::RealAccount(parsed),
                        ) => {
                            assert_eq!(parsed.static_data.is_signer, expected.is_signer);
                            assert_eq!(parsed.static_data.is_writable, expected.is_writable);
                            assert_eq!(parsed.static_data.data_len, expected.data_len);
                            assert_eq!(parsed.static_data.key, expected.key);
                            assert_eq!(parsed.static_data.owner, expected.owner);
                            assert_eq!(parsed.static_data.lamports, expected.lamports);
                            assert_eq!(parsed.data(), expected_data);
                        }
                        (
                            TestAccount::Duplicate(expected_index),
                            AccountInInstruction::Dup(parsed_index),
                        ) => {
                            assert_eq!(parsed_index, expected_index as usize);
                        }
                        _ => panic!("Mismatched account types"),
                    }
                    iterator = next_iter;
                }
                NextAccount::Data(_) => panic!("Expected an account, found instruction data"),
            }
        }

        // Check instruction data
        match iterator.next() {
            NextAccount::Data(parsed_data) => assert_eq!(parsed_data, instruction_data.as_slice()),
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
            (vec![TestAccount::Real(account1, data1.clone())], vec![5, 6]),
            (
                vec![
                    TestAccount::Real(account1, data1.clone()),
                    TestAccount::Real(account2, data2.clone()),
                ],
                vec![7, 8, 9],
            ),
            (
                vec![
                    TestAccount::Real(account1, data1.clone()),
                    TestAccount::Duplicate(0),
                    TestAccount::Real(account2, data2.clone()),
                ],
                vec![10],
            ),
            (
                vec![
                    TestAccount::Real(account1, data1),
                    TestAccount::Real(account2, data2),
                    TestAccount::Real(account3, data3),
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
        let mut instruction = create_test_instruction(vec![], instruction_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next() {
            NextAccount::Data(data) => assert_eq!(data, instruction_data.as_slice()),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_single_account_then_instruction_data() {
        let (account, data) = create_test_account(true, true, vec![1, 2, 3, 4]);
        let instruction_data = vec![5, 6, 7, 8];
        let mut instruction = create_test_instruction(
            vec![TestAccount::Real(account, data)],
            instruction_data.clone(),
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next() {
            NextAccount::Account(AccountInInstruction::RealAccount(acc), next_iter) => {
                assert_eq!(acc.static_data.is_signer, 1);
                assert_eq!(acc.static_data.is_writable, 1);
                assert_eq!(acc.static_data.data_len, 4);

                match next_iter.next() {
                    NextAccount::Data(data) => assert_eq!(data, instruction_data.as_slice()),
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
        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1),
                TestAccount::Real(account2, data2),
            ],
            instruction_data.clone(),
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        for _ in 0..2 {
            match iterator.next() {
                NextAccount::Account(AccountInInstruction::RealAccount(_), next_iter) => {
                    iterator = next_iter;
                }
                _ => panic!("Expected a real account"),
            }
        }

        match iterator.next() {
            NextAccount::Data(data) => assert_eq!(data, instruction_data.as_slice()),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_dup_account() {
        let (account, data) = create_test_account(false, false, vec![1, 2, 3, 4]);
        let mut instruction = create_test_instruction(
            vec![TestAccount::Duplicate(0), TestAccount::Real(account, data)],
            vec![],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next() {
            NextAccount::Account(AccountInInstruction::Dup(index), next_iter) => {
                assert_eq!(index, 0);
                iterator = next_iter;
            }
            _ => panic!("Expected a dup account"),
        }

        match iterator.next() {
            NextAccount::Account(AccountInInstruction::RealAccount(_), _) => {}
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
                )
            })
            .collect();
        let mut instruction = create_test_instruction(accounts, vec![]);

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let mut count = 0;

        while let NextAccount::Account(_, next_iter) = iterator.next() {
            count += 1;
            iterator = next_iter;
        }

        assert_eq!(count, num_accounts);
    }

    #[test]
    fn test_instruction_length_reading() {
        let instruction_data = vec![1, 2, 3, 4, 5];
        let mut instruction = create_test_instruction(vec![], instruction_data.clone());

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        match iterator.next() {
            NextAccount::Data(data) => assert_eq!(data.len(), instruction_data.len()),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_account_data_offset() {
        let (account1, data1) = create_test_account(false, false, vec![1, 2, 3, 4]);
        let (account2, data2) = create_test_account(false, false, vec![5, 6, 7, 8, 9, 10]);
        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1),
                TestAccount::Real(account2, data2),
            ],
            vec![],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        if let NextAccount::Account(AccountInInstruction::RealAccount(acc1), next_iter) =
            iterator.next()
        {
            assert_eq!(acc1.static_data.data_len, 4);
            iterator = next_iter;

            if let NextAccount::Account(AccountInInstruction::RealAccount(acc2), _) =
                iterator.next()
            {
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
        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1.clone()),
                TestAccount::Real(account2, data2.clone()),
            ],
            vec![],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        if let NextAccount::Account(AccountInInstruction::RealAccount(acc1), next_iter) =
            iterator.next()
        {
            assert_eq!(acc1.data(), data1);
            iterator = next_iter;

            if let NextAccount::Account(AccountInInstruction::RealAccount(acc2), _) =
                iterator.next()
            {
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
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data.clone())], vec![]);

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
    fn test_typed_known_next_full_account() {
        let test_data = TestStruct { a: 1, b: 2 };
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                &test_data as *const _ as *const u8,
                std::mem::size_of::<TestStruct>(),
            )
            .to_vec()
        };

        let (account, _) = create_test_account(true, true, data_bytes);
        let mut instruction = create_test_instruction(
            vec![TestAccount::Real(account, unsafe {
                std::slice::from_raw_parts(
                    &test_data as *const _ as *const u8,
                    std::mem::size_of::<TestStruct>(),
                )
                .to_vec()
            })],
            vec![],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.typed_known_next_full_account::<TestStruct>() };

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
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data.clone())], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.data_len as usize, data.len());
    }

    #[test]
    fn test_known_next_full_account_unaligned_size() {
        let data = vec![1, 2, 3, 4, 5]; // size % 8 != 0
        let (account, _) = create_test_account(true, false, data.clone());
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data.clone())], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.data_len as usize, data.len());
    }

    #[test]
    fn test_known_next_full_account_zero_size() {
        let data = vec![]; // zero size
        let (account, _) = create_test_account(true, false, data.clone());
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data.clone())], vec![]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.data_len as usize, data.len());
    }

    #[test]
    fn test_typed_known_next_full_account_aligned() {
        let aligned_data = AlignedStruct { a: 1, b: 2 };
        let mut instruction = create_typed_test_instruction(&aligned_data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.typed_known_next_full_account::<AlignedStruct>() };

        let data_as_struct = unsafe { &*(acc.data_ptr() as *const AlignedStruct) };
        assert_eq!(data_as_struct.a, 1);
        assert_eq!(data_as_struct.b, 2);
        assert_eq!(acc.data().len(), std::mem::size_of::<AlignedStruct>());
    }

    #[test]
    fn test_typed_known_next_full_account_unaligned() {
        let unaligned_data = UnalignedStruct { a: 1, b: 2 };
        let mut instruction = create_typed_test_instruction(&unaligned_data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.typed_known_next_full_account::<UnalignedStruct>() };

        let data_as_struct = unsafe { &*(acc.data_ptr() as *const UnalignedStruct) };
        assert_eq!(data_as_struct.a, 1);
        assert_eq!(data_as_struct.b, 2);
        assert_eq!(acc.data().len(), std::mem::size_of::<UnalignedStruct>());
    }

    #[test]
    fn test_typed_known_next_full_account_empty() {
        let empty_data = EmptyStruct {};
        let mut instruction = create_typed_test_instruction(&empty_data);
        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, _) = unsafe { iterator.typed_known_next_full_account::<EmptyStruct>() };

        assert_eq!(acc.data().len(), 0);
        assert_eq!(acc.data().len(), std::mem::size_of::<EmptyStruct>());
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

    // ==================== NEW TESTS ====================

    #[test]
    fn test_aligned_known_next_full_account() {
        // Test with 8-byte aligned data (multiple of 8)
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 8 bytes - aligned
        let (account, _) = create_test_account(true, false, data.clone());
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data.clone())], vec![9, 10]);

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc, next_iter) = unsafe { iterator.aligned_known_next_full_account() };

        assert_eq!(acc.data(), data.as_slice());
        assert_eq!(acc.static_data.is_signer, 1);
        assert_eq!(acc.static_data.is_writable, 0);
        assert_eq!(acc.static_data.data_len, 8);

        // Verify we can continue to instruction data
        match next_iter.next() {
            NextAccount::Data(instr_data) => assert_eq!(instr_data, &[9, 10]),
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
        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1.clone()),
                TestAccount::Real(account2, data2.clone()),
            ],
            vec![],
        );

        let iterator = unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
        let (acc1, next_iter) = unsafe { iterator.aligned_known_next_full_account() };
        assert_eq!(acc1.data(), data1.as_slice());

        let (acc2, next_iter) = unsafe { next_iter.aligned_known_next_full_account() };
        assert_eq!(acc2.data(), data2.as_slice());

        match next_iter.next() {
            NextAccount::Data(instr_data) => assert_eq!(instr_data, &[]),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_known_instruction_data() {
        let instruction_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (account, data) = create_test_account(true, true, vec![10, 20, 30]);
        let mut instruction = create_test_instruction(
            vec![TestAccount::Real(account, data)],
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
        let mut instruction = create_test_instruction(vec![], instruction_data.clone());

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
            vec![TestAccount::Real(account, data)],
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
            vec![TestAccount::Real(account, data)],
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
        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(account1, data1),
                TestAccount::Real(account2, data2),
                TestAccount::Real(account3, data3),
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
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data)], vec![5, 6]);

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
                let (acc, data) = create_test_account(i % 2 == 0, i % 3 == 0, vec![(i & 0xFF) as u8]);
                TestAccount::Real(acc, data)
            })
            .collect();

        let instruction_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let mut instruction = create_test_instruction(accounts, instruction_data.clone());

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
        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(account, data),
                TestAccount::Duplicate(0),
                TestAccount::Duplicate(0),
                TestAccount::Duplicate(0),
            ],
            vec![99],
        );

        let mut iterator =
            unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };

        // First should be real
        match iterator.next() {
            NextAccount::Account(AccountInInstruction::RealAccount(acc), next) => {
                assert_eq!(acc.data(), &[1, 2, 3, 4]);
                iterator = next;
            }
            _ => panic!("Expected real account"),
        }

        // Next three should be dups pointing to index 0
        for _ in 0..3 {
            match iterator.next() {
                NextAccount::Account(AccountInInstruction::Dup(idx), next) => {
                    assert_eq!(idx, 0);
                    iterator = next;
                }
                _ => panic!("Expected dup account"),
            }
        }

        // Finally instruction data
        match iterator.next() {
            NextAccount::Data(instr_data) => assert_eq!(instr_data, &[99]),
            _ => panic!("Expected instruction data"),
        }
    }

    #[test]
    fn test_dups_with_different_indices() {
        // Multiple real accounts, then dups pointing to different indices
        let (account0, data0) = create_test_account(true, false, vec![10]);
        let (account1, data1) = create_test_account(false, true, vec![20]);
        let (account2, data2) = create_test_account(true, true, vec![30]);

        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(account0, data0),
                TestAccount::Real(account1, data1),
                TestAccount::Real(account2, data2),
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
            match iterator.next() {
                NextAccount::Account(AccountInInstruction::RealAccount(acc), next) => {
                    assert_eq!(acc.data(), &[expected_data]);
                    iterator = next;
                }
                _ => panic!("Expected real account"),
            }
        }

        // Verify dup indices
        for expected_idx in [2usize, 0usize, 1usize] {
            match iterator.next() {
                NextAccount::Account(AccountInInstruction::Dup(idx), next) => {
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
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data.clone())], vec![]);

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
            let mut instruction =
                create_test_instruction(vec![TestAccount::Real(account, data.clone())], vec![]);

            let iterator =
                unsafe { AccountIterator::new_from_instruction(instruction.as_mut_ptr()) };
            let (acc, _) = unsafe { iterator.known_next_full_account() };

            assert_eq!(acc.static_data.lamports, lamports);
        }
    }

    #[test]
    fn test_mutation_persistence() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (account, _) = create_test_account(true, true, data.clone());
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, data)], vec![]);

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
        let mut instruction = create_test_instruction(
            vec![TestAccount::Real(account, data)],
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
        let mut instruction =
            create_test_instruction(vec![TestAccount::Real(account, large_data.clone())], vec![42]);

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
        assert_eq!(typed_acc.static_data.data_len, std::mem::size_of::<MyAccount>() as u64);

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
                TestAccount::Real(acc, data_bytes)
            })
            .collect();

        let mut instruction = create_test_instruction(accounts, vec![42]);

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

        let mut instruction = create_test_instruction(
            vec![
                TestAccount::Real(authority_acc, authority_data),
                TestAccount::Real(token_acc, token_bytes),
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
}
