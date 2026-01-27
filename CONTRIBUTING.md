# Contributing to BAHA

Thank you for your interest in contributing to BAHA! This document outlines the process for contributing to this project.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Community Guidelines](#community-guidelines)

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/baha.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up your development environment (see README.md for dependencies)

## Development Workflow

1. Make your changes in your feature branch
2. Ensure all tests pass
3. Add new tests if you're adding new functionality
4. Update documentation as needed
5. Submit a pull request to the main repository

## Code Standards

- Follow C++17 standards
- Use consistent formatting (prefer clang-format)
- Write clear, descriptive variable and function names
- Document public APIs with Doxygen-style comments
- Keep functions focused and reasonably sized
- Use RAII and smart pointers where appropriate

### Naming Conventions
- Classes and structs: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_member_name`

### Code Structure
- Header files should have include guards
- Keep headers lightweight
- Use forward declarations when possible
- Separate interface from implementation

## Testing

- All new functionality should include unit tests
- Bug fixes should include regression tests
- Tests should be in the `tests/` directory
- Use Google Test framework for C++ tests

Run tests with:
```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make
make test
```

## Pull Request Process

1. Ensure your PR addresses a single issue or adds a single feature
2. Update the README.md if needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Describe your changes in the PR description
6. Link to any relevant issues
7. Request review from maintainers

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Welcome new contributors
- Focus on the code, not the person
- Assume good faith

## Questions?

If you have questions about contributing, feel free to open an issue or contact the maintainers.

Thank you for contributing to BAHA!