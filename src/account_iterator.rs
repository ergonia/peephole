use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};
use solana_program::entrypoint::{MAX_PERMITTED_DATA_INCREASE, NON_DUP_MARKER};
use solana_program::pubkey::Pubkey;
use solana_sdk::entrypoint::BPF_ALIGN_OF_U128;

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
    const ALIGNMENT: Option<usize>;
    fn get_account_size(size_in_data: &u64) -> u64;
}

struct Dynamic;

impl Slurper for Dynamic {
    fn get_account_size(size_in_data: &u64) -> u64 {
        *size_in_data
    }

    const ALIGNMENT: Option<usize> = None;
}

struct TypedSlurper<T>(PhantomData<T>);

impl<T: Copy> Slurper for TypedSlurper<T> {
    fn get_account_size(size_in_data: &u64) -> u64 {
        let known_size = std::mem::size_of::<T>() as u64;
        unsafe {
            assume!(known_size == *size_in_data, "Known size is not real size");
        }
        known_size as u64
    }

    const ALIGNMENT: Option<usize> = Some(std::mem::align_of::<T>());
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
                let (non_dup, next) = self.slurp_real_account::<Dynamic>();
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
    pub unsafe fn typed_known_next_full_account<T: Copy>(
        self,
    ) -> (NonDupAccount<'static>, AccountIterator) {
        debug_assert!(self.remaining_accounts > 0);

        let is_dup = unsafe { &*self.base_ptr };

        debug_assert_eq!(*is_dup, NON_DUP_MARKER);

        let (account, next) = self.slurp_real_account::<TypedSlurper<T>>();

        (account, unsafe {
            AccountIterator::new_from_raw(next, self.remaining_accounts - 1)
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
    fn slurp_real_account<S: Slurper>(&self) -> (NonDupAccount<'static>, *mut u8) {
        let (account_static, next) = unsafe { slurp::<NonDupAccountStatic>(self.base_ptr) };
        unsafe {
            let data_len = S::get_account_size(&account_static.data_len);
            let data = std::slice::from_raw_parts_mut(next, data_len as usize);

            let next = next.add(account_static.data_len as usize + MAX_PERMITTED_DATA_INCREASE);

            let next = match S::ALIGNMENT {
                Some(alignment) if alignment % BPF_ALIGN_OF_U128 == 0 => {
                    debug_assert_eq!(next.align_offset(BPF_ALIGN_OF_U128), 0);
                    next
                }
                _ => next.add(next.align_offset(BPF_ALIGN_OF_U128)),
            };

            let (rent_epoch, next) = slurp::<u64>(next);

            let account = NonDupAccount {
                static_data: account_static,
                all_data: data,
                rent_epoch,
            };

            (account, next)
        }
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

#[cfg(test)]
mod tests {
    use crate::bytes::PodUtils;

    use super::*;
    use quickcheck::Arbitrary;
    use solana_program::pubkey::Pubkey;

    #[derive(Clone, Debug)]
    enum TestAccount {
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

    fn create_test_account(
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

    fn create_test_instruction(accounts: Vec<TestAccount>, instruction_data: Vec<u8>) -> Vec<u8> {
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
}
