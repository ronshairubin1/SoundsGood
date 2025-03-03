# Changelog

## Version 0.9

### Sound Directory Consolidation
- Consolidated all sound files into a single directory (`data/sounds/training_sounds/`)
- Removed redundant `Config.SOUNDS_DIR` variable and updated all code to directly reference `Config.TRAINING_SOUNDS_DIR`
- Modified the verification endpoint to prevent duplicate file copies
- Added a symbolic link at the old location (`sounds/`) for backward compatibility
- Created migration and cleanup scripts to safely handle the transition
- Updated documentation to reflect the new directory structure


### Final Cleanup
- Moved `src/static/goodsounds/` directory to `legacy/goodsounds/` as part of sound directory consolidation
- Moved symlink from `sounds/` to `legacy/sounds/` for historical reference
- Removed `sounds_backup` directory
- Eliminated backward compatibility for direct references to `sounds/`
- Moved migration scripts to `legacy/migrate/` directory as they're no longer needed for day-to-day operations

### Sound Directory Structure Enhancement
- Renamed `temp_sounds` to `pending_verification_live_recording_sounds` for clarity
- Added new dedicated directory `uploaded_sounds` for temporarily uploaded files during prediction
- Created `test_sounds` directory for development test recordings
- Updated all code to use the new directory structure directly with no aliases
- Added detailed documentation on the sound directory structure

### Benefits
- Eliminated duplicate storage of sound files
- Simplified the codebase and configuration by removing layer of indirection
- Maintained backward compatibility through symbolic links
- Improved maintainability by having a single source of truth for sound files 