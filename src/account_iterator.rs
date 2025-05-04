use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};
use solana_sdk::entrypoint::{BPF_ALIGN_OF_U128, MAX_PERMITTED_DATA_INCREASE, NON_DUP_MARKER};
use solana_sdk::pubkey::Pubkey;

use crate::{assume, bytes::slurp};

pub enum AccountInInstruction {
    RealAccount(NonDupAccount<'static>),
    Dup(usize),
}

pub enum NextAccount {
    Data(&'static mut [u8]),
    Account(AccountInInstruction, AccountIterator),
}

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
    /// The caller must ensure that the pointer is valid and the account count is correct.
    ///
    /// # Arguments
    ///
    /// * `base` - A pointer to the start of the instruction data.
    ///
    /// # Returns
    ///
    /// A new AccountIterator instance.
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
    /// The caller must ensure that the pointer is valid and the account count is correct.
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

    /// Retrieves the next full (non-duplicate) account as a typed account.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes the next account is a full account
    /// and that the account exactly holds one of the type passed with no extra allocated data.
    ///
    /// The caller must ensure that this assumption holds true.
    ///
    /// # Returns
    ///
    /// A tuple containing the next full account and the updated iterator.
    #[inline]
    pub unsafe fn static_slurp_typed_account<T: Pod + Zeroable>(
        self,
    ) -> (&'static mut TypedNonDupAccount<T>, AccountIterator) {
        let (single_account, next) = self.static_slurp_typed_accounts::<T, 1>();
        (&mut single_account[0], next)
    }

    /// Retrieves the next N full (non-duplicate) accounts as a typed array.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes the next N accounts are full accounts
    /// and that each account exactly holds one of the type passed with no extra allocated data.
    ///
    /// The caller must ensure that this assumption holds true.
    ///
    /// # Returns
    ///
    /// A tuple containing the next N full accounts as an array and the updated iterator.
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

    /// Retrieves the instruction data.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes the iterator is at the instruction data.
    /// The caller must ensure that all accounts have been processed before calling this.
    ///
    /// # Returns
    ///
    /// A slice containing the instruction data.
    #[inline]
    pub unsafe fn known_instruction_data(self) -> &'static mut [u8] {
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

        assert_eq!(
            std::mem::size_of::<T>() % 8,
            0,
            "Account size must be a multiple of 8"
        );
        assert_eq!(
            std::mem::size_of::<[TypedNonDupAccount<T>; N]>() % 8,
            0,
            "Account size must be less than 8 bytes"
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Eq, Debug)]
pub struct NonDupAccountStatic {
    pub is_dup: u8,
    pub is_signer: u8,
    pub is_writable: u8,
    pub executable: u8,
    pub original_data_len: u32,
    pub key: Pubkey,
    pub owner: Pubkey,
    pub lamports: u64,
    pub data_len: u64,
}

#[derive(PartialEq, Eq)]
pub struct NonDupAccount<'a> {
    pub static_data: &'a NonDupAccountStatic,
    pub all_data: &'a mut [u8],
    pub rent_epoch: &'a u64,
}

#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(C)]
pub struct TypedNonDupAccount<T: Pod + Zeroable> {
    pub static_data: NonDupAccountStatic,
    pub data: T,
    _buffer: [u8; MAX_PERMITTED_DATA_INCREASE],
    pub rent_epoch: u64,
}

unsafe impl<T: Pod + Zeroable> Pod for TypedNonDupAccount<T> {}
unsafe impl<T: Pod + Zeroable> Zeroable for TypedNonDupAccount<T> {}

impl<'a> NonDupAccount<'a> {
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

    use solana_sdk::pubkey::Pubkey;
    use std::rc::Rc;

    use crate::bytes::PodUtils;

    use super::*;
    use quickcheck::Arbitrary;
    use solana_sdk::entrypoint;

    #[derive(Clone, Debug)]
    pub enum TestAccount {
        Real(NonDupAccountStatic, Vec<u8>),
        Duplicate(u8),
    }

    fn arbitrary_array<T: Arbitrary>(g: &mut quickcheck::Gen) -> [T; 32] {
        std::array::from_fn(|_| T::arbitrary(g))
    }

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
                    key: Pubkey::new_from_array(arbitrary_array(g)),
                    owner: Pubkey::new_from_array(arbitrary_array(g)),
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
        let account = NonDupAccountStatic {
            is_dup: NON_DUP_MARKER,
            is_signer: is_signer as u8,
            is_writable: is_writable as u8,
            executable: 0,
            original_data_len: data.len() as u32,
            key: Pubkey::new_unique(),
            owner: Pubkey::new_unique(),
            lamports: 100,
            data_len: data.len() as u64,
        };
        (account, data)
    }

    pub fn create_test_instruction(
        accounts: Vec<TestAccount>,
        instruction_data: Vec<u8>,
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

                    instruction.extend_from_slice(&0u64.to_le_bytes()); // rent_epoch
                }
                TestAccount::Duplicate(index) => {
                    instruction.extend_from_slice(&(index as u64).to_le_bytes());
                }
            }
        }

        let instruction_len = instruction_data.len() as u64;
        instruction.extend_from_slice(&instruction_len.to_le_bytes());
        instruction.extend_from_slice(&instruction_data);

        instruction
    }

    // Typed account tests
    #[derive(Copy, Clone)]
    #[repr(C)]
    pub struct AlignedStruct {
        // 8-byte aligned
        pub a: u64,
        pub b: u64,
    }

    #[derive(Copy, Clone)]
    #[repr(C)]
    pub struct UnalignedStruct {
        // Not 8-byte aligned
        pub a: u16,
        pub b: u8,
    }

    #[derive(Copy, Clone)]
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

    #[derive(Debug, Clone)]
    pub enum TestAccountType {
        Aligned(QuickCheckAligned, bool, bool),
        Unaligned(QuickCheckUnaligned, bool, bool),
        Empty(QuickCheckEmpty, bool, bool),
        Untyped(Vec<u8>, bool, bool),
        AlignedArray([(QuickCheckAligned, bool, bool); 2]),
        AlignedArray3([(QuickCheckAligned, bool, bool); 3]),
    }

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
            match u8::arbitrary(g) % 6 {
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
                _ => TestAccountType::AlignedArray3([aligned(g), aligned(g), aligned(g)]),
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
        }
    }

    pub fn do_quickcheck_mixed_account_types_with_arrays(
        account_types: Vec<TestAccountType>,
        instruction_data_gen: Vec<u8>,
    ) -> bool {
        if account_types.is_empty() {
            return true;
        }

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
        // 1. Generate TestAccount structures from TestAccountType
        let test_accounts: Vec<_> = account_types
            .iter()
            .flat_map(|t| {
                let accounts = create_account_from_type(t.clone());
                accounts
                    .into_iter()
                    .map(|(acc, data)| TestAccount::Real(acc, data))
            })
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
            unsafe { entrypoint::deserialize(instruction_buffer.as_mut_ptr()) };

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
                    // Check if Solana account is *not* a duplicate derived one.
                    // Solana's deserialize clones AccountInfo for duplicates. We rely on Rc ptr equality
                    // to differentiate original from cloned duplicates for this check.
                    // If it's not the first account, check it wasn't cloned from a previous one.
                    let is_solana_original = if i > 0 {
                        let mut found_clone = false;
                        for j in 0..i {
                            if Rc::ptr_eq(&accounts_solana[j].lamports, &solana_acc.lamports)
                                && Rc::ptr_eq(&accounts_solana[j].data, &solana_acc.data)
                                && accounts_solana[j].key == solana_acc.key
                            // Key comparison as extra safety
                            {
                                found_clone = true;
                                break;
                            }
                        }
                        !found_clone
                    } else {
                        true // First account is always original if present
                    };

                    if !is_solana_original {
                        eprintln!(
                            "Account type mismatch at index {}: Fast=Real, Solana=Duplicate",
                            i
                        );
                        return false;
                    }

                    // Compare fields
                    if fast_real.static_data.key != *solana_acc.key {
                        eprintln!("Key mismatch at index {}", i);
                        return false;
                    }
                    if (fast_real.static_data.is_signer != 0) != solana_acc.is_signer {
                        eprintln!("is_signer mismatch at index {}", i);
                        return false;
                    }
                    if (fast_real.static_data.is_writable != 0) != solana_acc.is_writable {
                        eprintln!("is_writable mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.owner != *solana_acc.owner {
                        eprintln!("Owner mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.lamports != **(solana_acc.lamports.borrow()) {
                        eprintln!("Lamports mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.data_len != solana_acc.data.borrow().len() as u64 {
                        eprintln!("Data length mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.data() != *solana_acc.data.borrow() {
                        eprintln!("Data mismatch at index {}", i);
                        return false;
                    }
                    if fast_real.static_data.executable != solana_acc.executable as u8 {
                        eprintln!("Executable mismatch at index {}", i);
                        return false;
                    }
                    if *fast_real.rent_epoch != solana_acc.rent_epoch {
                        eprintln!("Rent epoch mismatch at index {}", i);
                        return false;
                    }
                }
                AccountInInstruction::Dup(fast_dup_index) => {
                    // Check if Solana account *is* a duplicate by checking Rc ptr equality
                    let original_solana_acc = &accounts_solana[*fast_dup_index];
                    if !Rc::ptr_eq(&original_solana_acc.lamports, &solana_acc.lamports)
                        || !Rc::ptr_eq(&original_solana_acc.data, &solana_acc.data)
                        || original_solana_acc.key != solana_acc.key
                    // Sanity check key too
                    {
                        eprintln!("Account type mismatch at index {}: Fast=Duplicate({}), Solana=Real or wrong duplicate", i, fast_dup_index);
                        return false;
                    }
                    // No need to compare fields further, Rc::ptr_eq confirms it's a clone of the correct original
                }
            }
        }

        true // All checks passed
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::bytes::PodUtils;

    use super::*;
    use quickcheck::Arbitrary;
    use solana_sdk::entrypoint;

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
}
