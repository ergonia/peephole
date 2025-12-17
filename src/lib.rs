#![no_std]
#![allow(unexpected_cfgs)]

//! # peephole
//!
//! Zero-copy account parsing for Solana programs.
//!
//! Standard deserialization allocates and copies. peephole gives you typed pointers
//! directly into the runtime's buffer—reads and writes happen in-place.
//!
//! Real-world performance: [51 CU oracle update](https://solscan.io/tx/JkGJc3Q2eAjPKWG4PjbNxYguqc6gAqD96bMaAmqrotH2dn5cWXV7w8Sgp7tbckr1QKqab6749rhgPjTnQEDwDkB),
//! [67 CUs for 3 oracle updates](https://solscan.io/tx/4w5T3BrUUb2zmcuVwZjNaiHY2ysfMeMNa5NE9bHfoywc8iPTJCfcFndZ2C9TyGQTj3jLMaVRALbRDDpWV9HAEHhU),
//! [12 CU trading contract](https://solscan.io/tx/3ggc3ZbGQ1Zop9JPHSS9bBa7pZ1i5fwifkwJ7ALNehTSjRcVySsAs4YgZFrkeK8NvULJEm54XCDXDGNu2zor6Y2Q), and many others.
//!
//! ## Iteration
//!
//! ```ignore
//! use peephole::account_iterator::{AccountIterator, NextHeader};
//!
//! let mut iter = unsafe { AccountIterator::new_from_instruction(input) };
//! loop {
//!     match iter.next_header() {
//!         NextHeader::Header(cursor) => {
//!             // Inspect header before parsing data
//!             let key = &cursor.static_data.key;
//!             let data_len = cursor.data_len();
//!             let (acc, next) = unsafe { cursor.parse_data() };
//!             iter = next;
//!         }
//!         NextHeader::Dup(idx, next) => {
//!             // Duplicate of account `idx`
//!             iter = next;
//!         }
//!         NextHeader::Data(instruction_data, program_id) => {
//!             // instruction_data: &mut [u8], program_id: &Pubkey
//!             break;
//!         }
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
//! - [`assume!`]: `debug_assert!` that becomes `unreachable_unchecked` in release
//! - [`pubkey_byte_map`]: O(1) pubkey lookup (up to 256 keys, indexed by first byte)

#[cfg(any(test, feature = "std"))]
extern crate std;

pub mod account_iterator;
pub mod assume;
pub mod bytes;
pub mod pubkey_byte_map;
pub mod solana_export;
pub mod utils;
