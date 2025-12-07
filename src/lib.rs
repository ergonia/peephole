//! # peephole
//!
//! Zero-copy account parsing for Solana programs.
//!
//! Standard deserialization allocates and copies. peephole gives you typed pointers
//! directly into the runtime's buffer—reads and writes happen in-place.
//!
//! Real-world performance: [51 CU oracle update](https://solscan.io/tx/JkGJc3Q2eAjPKWG4PjbNxYguqc6gAqD96bMaAmqrotH2dn5cWXV7w8Sgp7tbckr1QKqab6749rhgPjTnQEDwDkB),
//! [67 CU for 3 oracle updates](https://solscan.io/tx/4w5T3BrUUb2zmcuVwZjNaiHY2ysfMeMNa5NE9bHfoywc8iPTJCfcFndZ2C9TyGQTj3jLMaVRALbRDDpWV9HAEHhU).
//!
//! ## Iteration
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
//!             iter = next;
//!         }
//!         NextAccount::Data(instruction_data) => break,
//!     }
//! }
//! ```
//!
//! ## Typed Slurping
//!
//! When you know the account's data layout, cast directly to your struct:
//!
//! ```ignore
//! let (account, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };
//! account.data.amount += 100;
//!
//! // Batch slurp N accounts of the same type
//! let (accounts, iter) = unsafe { iter.static_slurp_typed_accounts::<TokenAccount, 3>() };
//! ```
//!
//! ## Safety
//!
//! Unsafe methods assume correct layout. Verify authority first, then trust the rest:
//!
//! ```ignore
//! let (authority, iter) = unsafe { iter.known_next_full_account() };
//! if !(is_authority(&authority.static_data.key) && authority.static_data.is_signer()) {
//!     return Err(Unauthorized);
//! }
//! // Authority verified—safe to trust remaining account types
//! let (token, iter) = unsafe { iter.static_slurp_typed_account::<TokenAccount>() };
//! ```
//!
//! Debug builds check invariants (account counts, no unexpected duplicates, size matches).
//! Release builds trust you completely.
//!
//! ## Features
//!
//! `solana-sdk` or `pinocchio-sdk` (exactly one required).
//!
//! ## Modules
//!
//! - [`account_iterator`]: Core iterator and account types ([`AccountIterator`](account_iterator::AccountIterator),
//!   [`NonDupAccount`](account_iterator::NonDupAccount), [`TypedNonDupAccount`](account_iterator::TypedNonDupAccount))
//! - [`assume`]: `debug_assert!` that becomes `unreachable_unchecked` in release
//! - [`pubkey_byte_map`]: O(1) pubkey lookup by first byte for known key sets
//!

#![allow(unexpected_cfgs)]

pub mod account_iterator;
pub mod assume;
pub mod bytes;
pub mod pubkey_byte_map;
pub mod solana_export;
pub mod utils;
