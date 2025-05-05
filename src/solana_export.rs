#[cfg(all(feature = "solana-sdk", feature = "pinocchio-sdk"))]
compile_error!(
    "Features 'solana-sdk' and 'pinocchio-sdk' are mutually exclusive. Please enable only one."
);

#[cfg(not(any(feature = "solana-sdk", feature = "pinocchio-sdk")))]
compile_error!("An SDK feature ('solana-sdk' or 'pinocchio-sdk') must be enabled.");

#[cfg(any(test, fuzzing))]
pub trait IsAccount {
    type Pubkey;
    fn get_key(&self) -> &Self::Pubkey;
    fn get_is_signer(&self) -> bool;
    fn get_is_writable(&self) -> bool;
    fn get_owner(&self) -> &Self::Pubkey;
    fn get_lamports(&self) -> u64;
    fn get_data_len(&self) -> usize;
    fn get_data(&self) -> &[u8];
    fn get_executable(&self) -> bool;
}

#[cfg(feature = "solana-sdk")]
mod solana {
    pub use solana_program::*;

    pub mod constants {
        pub use super::entrypoint::{
            BPF_ALIGN_OF_U128, MAX_PERMITTED_DATA_INCREASE, NON_DUP_MARKER,
        };
    }

    #[cfg(any(test, fuzzing))]
    use super::IsAccount;

    #[inline]
    pub const fn pubkey_from_array(array: [u8; 32]) -> pubkey::Pubkey {
        pubkey::Pubkey::new_from_array(array)
    }

    #[inline]
    pub const fn pubkey_bytes(pubkey: &pubkey::Pubkey) -> [u8; 32] {
        pubkey.to_bytes()
    }

    pub fn unique_pubkey() -> pubkey::Pubkey {
        pubkey::Pubkey::new_unique()
    }

    #[cfg(any(test, fuzzing))]
    pub fn easy_deserialize<'a>(
        inputs: *mut u8,
    ) -> (pubkey::Pubkey, Vec<account_info::AccountInfo<'a>>, Vec<u8>) {
        let (program_id, accounts, instruction_data) = unsafe { entrypoint::deserialize(inputs) };

        (*program_id, accounts, instruction_data.to_vec())
    }

    #[cfg(any(test, fuzzing))]
    impl IsAccount for account_info::AccountInfo<'_> {
        type Pubkey = pubkey::Pubkey;
        fn get_key(&self) -> &Self::Pubkey {
            self.key
        }
        fn get_is_signer(&self) -> bool {
            self.is_signer
        }
        fn get_is_writable(&self) -> bool {
            self.is_writable
        }
        fn get_owner(&self) -> &Self::Pubkey {
            self.owner
        }
        fn get_lamports(&self) -> u64 {
            **self.lamports.borrow()
        }
        fn get_data_len(&self) -> usize {
            self.data_len()
        }
        fn get_data(&self) -> &[u8] {
            let borrow = self.try_borrow_data().unwrap();
            Vec::leak(borrow.to_vec())
        }
        fn get_executable(&self) -> bool {
            self.executable
        }
    }
}

#[cfg(feature = "pinocchio-sdk")]
mod solana {

    use std::sync::atomic::{AtomicU64, Ordering};

    pub use pinocchio::*;

    pub mod constants {
        pub use pinocchio::account_info::MAX_PERMITTED_DATA_INCREASE;
        /// `assert_eq(core::mem::align_of::<u128>(), 8)` is true for BPF but not
        /// for some host machines.
        pub const BPF_ALIGN_OF_U128: usize = 8;

        /// Value used to indicate that a serialized account is not a duplicate.
        pub const NON_DUP_MARKER: u8 = u8::MAX;
    }

    #[cfg(any(test, fuzzing))]
    use super::IsAccount;

    #[inline]
    pub const fn pubkey_from_array(array: [u8; 32]) -> pubkey::Pubkey {
        array
    }

    #[inline]
    pub const fn pubkey_bytes(pubkey: &pubkey::Pubkey) -> [u8; 32] {
        *pubkey
    }

    static PUBKEY_CTR: AtomicU64 = AtomicU64::new(0);

    pub fn unique_pubkey() -> pubkey::Pubkey {
        let ctr = PUBKEY_CTR.fetch_add(1, Ordering::Relaxed);
        let mut array = [0u8; 32];
        let ctr_bytes = ctr.to_le_bytes();
        array[..8].copy_from_slice(&ctr_bytes);
        pubkey_from_array(array)
    }

    #[cfg(any(test, fuzzing))]
    pub fn easy_deserialize(
        inputs: *mut u8,
    ) -> (pubkey::Pubkey, Vec<account_info::AccountInfo>, Vec<u8>) {
        use std::mem::MaybeUninit;

        let mut the_uninit = Vec::new();
        for _ in 0..256 {
            the_uninit.push(MaybeUninit::uninit());
        }
        let (program_id, n_accounts, instruction_data) =
            unsafe { entrypoint::deserialize::<256>(inputs, &mut the_uninit) };

        let real_accounts = the_uninit
            .into_iter()
            .take(n_accounts)
            .map(|x| unsafe { x.assume_init() })
            .collect();
        (*program_id, real_accounts, instruction_data.to_vec())
    }

    #[cfg(any(test, fuzzing))]
    impl IsAccount for account_info::AccountInfo {
        type Pubkey = pubkey::Pubkey;
        fn get_key(&self) -> &Self::Pubkey {
            self.key()
        }
        fn get_is_signer(&self) -> bool {
            self.is_signer()
        }
        fn get_is_writable(&self) -> bool {
            self.is_writable()
        }
        fn get_owner(&self) -> &Self::Pubkey {
            self.owner()
        }
        fn get_lamports(&self) -> u64 {
            self.lamports()
        }
        fn get_data_len(&self) -> usize {
            self.data_len()
        }
        fn get_data(&self) -> &[u8] {
            unsafe { self.borrow_data_unchecked() }
        }
        fn get_executable(&self) -> bool {
            self.executable()
        }
    }
}

pub use solana::*;
