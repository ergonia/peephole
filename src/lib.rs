//! # peephole
//!
//! Zero-copy account parsing for Solana programs.
//!
//! The standard `solana_program::entrypoint::deserialize` allocates `AccountInfo` structs
//! and copies data into them. `peephole` skips that—you get typed pointers directly into
//! the runtime's buffer, so reads and writes happen in-place.
//!
//! ## Basic Usage
//!
//! ```ignore
//! use peephole::account_iterator::{AccountIterator, NextAccount, AccountInInstruction};
//!
//! let mut iter = unsafe { AccountIterator::new_from_instruction(input) };
//! loop {
//!     match iter.next() {
//!         NextAccount::Account(AccountInInstruction::RealAccount(acc), next) => {
//!             let key = &acc.static_data.key;
//!             let data = acc.data();
//!             iter = next;
//!         }
//!         NextAccount::Account(AccountInInstruction::Dup(idx), next) => {
//!             // Duplicate of account at index `idx`
//!             iter = next;
//!         }
//!         NextAccount::Data(instruction_data, program_iter) => {
//!             // Access the program ID after instruction data
//!             let program_id = program_iter.program_address();
//!             break;
//!         }
//!     }
//! }
//! ```
//!
//! ## Typed Slurping
//!
//! Cast accounts directly to your struct (`T: Pod + Zeroable`, 8-byte aligned):
//!
//! ```ignore
//! let (account, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };
//! let amount = account.data.amount;  // direct field access
//!
//! // Batch slurp N accounts as an array
//! let (accounts, iter) = unsafe { iter.static_slurp_typed_accounts::<TokenAccount, 3>() };
//! ```
//!
//! ## Safety
//!
//! The unsafe methods assume you know the account layout. Typical pattern: verify a
//! trusted signer first, then use unsafe methods on the remaining accounts.
//!
//! Debug builds verify invariants. Release builds trust you.
//!
//! ## Feature Flags
//!
//! `solana-sdk` or `pinocchio-sdk` (exactly one required).
//!
//! ## Modules
//!
//! - [`account_iterator`]: The core iterator and account types
//! - [`assume`]: `debug_assert!` that becomes `unreachable_unchecked` in release
//! - [`pubkey_byte_map`]: O(1) pubkey lookup (up to 256 keys, indexed by first byte)
//!

#![allow(unexpected_cfgs)]

pub mod account_iterator;
pub mod assume;
pub mod bytes;
pub mod pubkey_byte_map;
pub mod solana_export;
pub mod utils;
