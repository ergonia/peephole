# Fast Instruction

`fast_instruction` is a high-performance Rust library designed to optimize and enhance Solana smart contract development. It provides a set of tools and utilities for efficient account and instruction handling, as well as optimized operations commonly used in Solana programs.

## Features

- **Account Iterator**: Efficient iteration over accounts in Solana instructions.
- **Assumption Macros**: Unsafe macros for performance-critical code sections.
- **Byte Manipulation**: Fast and safe byte operations for Solana's data structures.
- **Utility Functions**: Optimized functions for common Solana operations.

## Rationale

Solana's high-performance blockchain requires smart contracts to be as efficient as possible. Every compute unit counts, and even small optimizations can lead to significant improvements in throughput and cost-effectiveness. This library aims to provide developers with tools to squeeze out extra performance where it matters most.

## Modules

### `account_iterator`

Provides structures and methods for efficiently iterating over accounts in Solana instructions. This module optimizes memory access patterns and reduces overhead when processing multiple accounts.

**Use Case**: When your smart contract needs to process a variable number of accounts efficiently.

### `assume`

Contains macros for making performance-critical assumptions. These macros allow developers to inform the compiler about conditions that are always true, enabling more aggressive optimizations.

**Use Case**: When you have invariants in your code that the compiler can't deduce but you know are always true.

### `bytes`

Offers utilities for efficient byte manipulation and conversion. This module provides fast, safe operations on byte slices and structures, which are common in Solana's data model.

**Use Case**: When you need to perform low-level byte operations without sacrificing performance.

### `utils`

Includes utility functions like fast public key comparison. These functions are optimized versions of common operations in Solana programs.

**Use Case**: When you need to perform frequent comparisons of public keys or other common Solana-specific operations.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
fast-instruction = "0.1.0"
```

Here's a basic example with `AccountIterator`:

```rust
use fast_instruction::account_iterator::AccountIterator;

// Assuming `instruction_data` is a pointer to your instruction data
let mut iterator = AccountIterator::new_from_instruction(instruction_data);

while let NextAccount::Account(account, next_iter) = iterator.next() {
    // Process the account
    // ...
    iterator = next_iter;
}

// Handle instruction data
if let NextAccount::Data(data) = iterator.next() {
    // Process instruction data
    // ...
}
```

## Safety

This library contains unsafe code and should be used with caution. It's designed for performance-critical scenarios in Solana smart contract development. Ensure you understand the implications of using these optimizations before incorporating them into your project.

## Performance Considerations

- The `AccountIterator` provides a more efficient way to iterate over accounts compared to standard slices.
- Assumption macros can lead to significant performance improvements but must be used carefully to avoid undefined behavior.
- Byte manipulation utilities are optimized for common Solana data structures and operations.
- Utility functions like `fast_cmp_pubkey` offer small but meaningful performance gains in tight loops or frequently called code.
