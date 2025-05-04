//! # Fast Instruction
//!
//! `fast_instruction` is a Rust library designed to optimize and enhance the performance
//! of Solana smart contract development. It provides a set of tools and utilities for
//! efficient account and instruction handling, as well as optimized operations commonly
//! used in Solana programs.
//!
//! ## Features
//!
//! - **Account Iterator**: Efficient iteration over accounts in Solana instructions.
//! - **Assumption Macros**: Unsafe macros for performance-critical code sections.
//! - **Byte Manipulation**: Fast and safe byte operations for Solana's data structures.
//! - **Utility Functions**: Optimized functions for common Solana operations.
//!
//! ## Modules
//!
//! - [`account_iterator`]: Provides structures and methods for iterating over accounts.
//! - [`assume`]: Contains macros for making performance-critical assumptions.
//! - [`bytes`]: Offers utilities for efficient byte manipulation and conversion.
//! - [`utils`]: Includes utility functions like fast public key comparison.
//!
//! ## Safety
//!
//! This library contains unsafe code and should be used with caution. It's designed
//! for performance-critical scenarios in Solana smart contract development. Ensure
//! you understand the implications of using these optimizations before incorporating
//! them into your project.
//!
//! For more detailed information, refer to the documentation of individual modules
//! and functions.

pub mod account_iterator;
pub mod assume;
pub mod bytes;
pub mod pubkey_byte_map;
pub mod solana_export;
pub mod utils;
