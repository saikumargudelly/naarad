# Pydantic v2 Migration and Dependency Updates

This document outlines the changes made to update the project to use Pydantic v2 and modernize the dependency stack.

## Major Updates

### 1. Pydantic v2 Migration
- Upgraded from Pydantic v1 to v2 with all related packages
- Updated all Pydantic models to use new v2 syntax:
  - Replaced `class Config` with `model_config = ConfigDict(...)`
  - Updated validators to use `@field_validator` and `@model_validator`
  - Added proper type hints and field configurations
  - Implemented JSON encoders for datetime and other complex types

### 2. Dependency Updates
- Updated all dependencies to their latest stable versions compatible with Python 3.12
- Added new dependencies for better Pydantic v2 support:
  - `pydantic-extra-types` for additional field types
  - `email-validator` for email validation
- Removed deprecated or incompatible packages

### 3. Development Environment
- Updated development dependencies to latest stable versions
- Added new development tools for better type checking and code quality
- Improved test configurations for better compatibility

## How to Update Your Environment

1. **Backup your current environment** (recommended)
   ```bash
   pip freeze > requirements_backup.txt
   ```

2. **Update pip and setuptools**
   ```bash
   python -m pip install --upgrade pip setuptools
   ```

3. **Install the updated dependencies**
   ```bash
   # Make the update scripts executable
   chmod +x update_deps.sh update_dev_deps.sh
   
   # Update production dependencies
   ./update_deps.sh
   
   # Update development dependencies
   ./update_dev_deps.sh
   ```

4. **Verify the installation**
   ```bash
   python -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"
   ```

## Notable Changes in Pydantic v2

1. **Configuration**
   - Use `model_config` instead of `class Config`
   - New `ConfigDict` for type-safe configuration
   - Updated validation behavior

2. **Validators**
   - New `@field_validator` and `@model_validator` decorators
   - Simplified validation logic
   - Better error messages

3. **Performance**
   - Up to 5-50x faster model validation
   - Reduced memory usage
   - Better caching of validation results

## Troubleshooting

If you encounter any issues:

1. Check the [Pydantic v2 migration guide](https://docs.pydantic.dev/2.7/migration/)
2. Verify all custom validators are using the new syntax
3. Ensure all field types are properly annotated
4. Check for any third-party packages that might need updates

## Next Steps

1. Run the test suite to verify everything works as expected
2. Update any documentation or examples to reflect the new syntax
3. Monitor for any deprecation warnings during development
4. Consider enabling Pydantic's strict mode for new code

## Rollback Plan

If you need to rollback:

1. Restore your virtual environment from backup
2. Revert the changes using git
3. Reinstall the old dependencies:
   ```bash
   pip install -r requirements_backup.txt
   ```
