/// Assumes that a condition is true and triggers undefined behavior if it's false.
///
/// This macro is used for performance-critical code where a condition is known to be true,
/// but the compiler cannot prove it. In debug builds, it performs an assertion.
/// In release builds, it informs the compiler that the condition is always true.
///
/// # Safety
///
/// This macro is unsafe because it can lead to undefined behavior if the condition is false.
/// Use with extreme caution and only when you are absolutely certain the condition is true.
///
/// # Examples
///
/// ```
/// use fast_instruction::assume;
/// let x = 42;
/// unsafe {
///     assume!(x == 42, "x should always be 42");
/// }
/// // The compiler now knows that x is definitely 42
/// ```
#[macro_export]
macro_rules! assume {
    ($cond:expr, $rest:tt) => {
        let _value: bool = $cond;

        #[cfg(not(fuzzing))]
        debug_assert!(_value, $rest);

        if !_value {
            std::hint::unreachable_unchecked();
        }
    };
}

/// Assumes that an expression matches a specific pattern and triggers undefined behavior if it doesn't.
///
/// This macro is similar to `assume!`, but works with pattern matching. It's useful when you know
/// an expression will always match a certain pattern, but the compiler cannot prove it.
///
/// # Safety
///
/// This macro is unsafe because it can lead to undefined behavior if the expression doesn't match the pattern.
/// Use with extreme caution and only when you are absolutely certain about the match.
///
/// # Examples
///
/// ```
/// use fast_instruction::assume_matches;
/// enum MyEnum { Variant(i32) }
/// let x = MyEnum::Variant(42);
/// let result = unsafe {
///     assume_matches!(x, MyEnum::Variant(y) if y > 0 => y * 2)
/// };
/// assert_eq!(result, 84);
/// ```
#[macro_export]
macro_rules! assume_matches {
    ($expr:expr, $pattern:pat $(if $guard:expr)? => $result:expr) => {
        match $expr {
            $pattern $(if $guard)? => $result,
            _ => {
                let msg = stringify!($expr => $pattern $(if $guard)?);
                #[cfg(not(fuzzing))]
                debug_assert!(false, "{}", msg);
                std::hint::unreachable_unchecked()
            },
        }
    };

}

/// Assumes that a pointer has a specific offset from an initial pointer.
///
/// This macro is used to inform the compiler about pointer arithmetic that it cannot verify.
/// In debug builds, it checks the offset. In release builds, it assumes the offset is correct.
///
/// # Safety
///
/// This macro is unsafe because it can lead to undefined behavior if the actual offset doesn't match the expected one.
/// Use with extreme caution and only when you are absolutely certain about the pointer arithmetic.
///
/// # Examples
///
/// ```
/// use fast_instruction::assume_offset;
/// use std::ptr::NonNull;
/// let array = [1, 2, 3, 4, 5];
/// let ptr = NonNull::from(&array[0]);
/// let offset_ptr = unsafe { ptr.as_ptr().add(2) };
/// unsafe {
///     assume_offset!(offset_ptr, ptr.as_ptr(), 2, "Offset should be 2");
/// }
/// ```
#[macro_export]
macro_rules! assume_offset {
    ($ptr:expr, $init:expr, $expected:expr, $rest:tt) => {
        #[cfg(debug_assertions)]
        {
            let offset = $ptr.offset_from($init);
            #[cfg(not(fuzzing))]
            debug_assert_eq!(offset, $expected as isize, $rest);
        }
        if $ptr != $init.add($expected) {
            std::hint::unreachable_unchecked();
        }
    };
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_assume() {
        let x = 42;
        unsafe {
            assume!(x == 42, "x should be 42");
        }
        // If we reach here, the assumption was correct
    }

    #[test]
    fn test_assume_matches() {
        #[allow(dead_code)]
        enum TestEnum {
            A(i32),
            B(bool),
        }
        let value = TestEnum::A(42);
        let result = unsafe { assume_matches!(value, TestEnum::A(x) if x > 0 => x * 2) };
        assert_eq!(result, 84);
    }

    #[test]
    fn test_assume_offset() {
        let array = [1, 2, 3, 4, 5];
        let ptr = std::ptr::NonNull::from(&array[0]);
        let offset_ptr = unsafe { ptr.as_ptr().add(2) };
        unsafe {
            assume_offset!(offset_ptr, ptr.as_ptr(), 2, "Offset should be 2");
        }
        assert_eq!(unsafe { *offset_ptr }, 3);
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_assume_failure() {
        let x = 41;
        unsafe {
            assume!(x == 42, "This should fail in debug mode");
        }
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_assume_matches_failure() {
        #[allow(dead_code)]
        enum TestEnum {
            A(i32),
            B(bool),
        }
        let value = TestEnum::B(true);
        unsafe {
            assume_matches!(value, TestEnum::A(x) => x * 2);
        }
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_assume_offset_failure() {
        let array = [1, 2, 3, 4, 5];
        let ptr = std::ptr::NonNull::from(&array[0]);
        let offset_ptr = unsafe { ptr.as_ptr().add(2) };
        unsafe {
            assume_offset!(
                offset_ptr,
                ptr.as_ptr(),
                3,
                "This should fail in debug mode"
            );
        }
    }
}
