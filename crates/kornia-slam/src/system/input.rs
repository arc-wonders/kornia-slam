//! Validation of the input to [`SlamSystem::process_image`](super::SlamSystem::process_image)
//! and [`SlamSystem::process_stereo`](super::SlamSystem::process_stereo).

use kornia_image::{Image, ImageError, ImageSize};

/// A frame the system cannot process. The system state is unchanged.
#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    #[error("timestamp {timestamp_sec} is not finite")]
    NonFiniteTimestamp { timestamp_sec: f64 },
    #[error("the image is empty")]
    EmptyImage,
    #[error("the stereo pipeline needs a right image, use process_stereo")]
    StereoRequired,
    #[error(
        "right image is {}x{}, left is {}x{}",
        right.width, right.height, left.width, left.height
    )]
    StereoSizeMismatch { left: ImageSize, right: ImageSize },
    #[error("feature extraction failed: {0}")]
    Frontend(#[source] ImageError),
}

/// Checks the input before any system state is touched. `stereo` is whether
/// the pipeline selected stereo cameras; a right image for a mono pipeline is
/// ignored.
pub(crate) fn validate(
    timestamp_sec: f64,
    image: &Image<u8, 1>,
    right_image: Option<&Image<u8, 1>>,
    stereo: bool,
) -> Result<(), ProcessError> {
    if !timestamp_sec.is_finite() {
        return Err(ProcessError::NonFiniteTimestamp { timestamp_sec });
    }
    if is_empty(image.size()) {
        return Err(ProcessError::EmptyImage);
    }
    if !stereo {
        return Ok(());
    }
    let right = right_image.ok_or(ProcessError::StereoRequired)?;
    if right.size() != image.size() {
        return Err(ProcessError::StereoSizeMismatch {
            left: image.size(),
            right: right.size(),
        });
    }
    Ok(())
}

fn is_empty(size: ImageSize) -> bool {
    size.width == 0 || size.height == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize) -> Image<u8, 1> {
        Image::from_size_val(ImageSize { width, height }, 0).unwrap()
    }

    #[test]
    fn mono_input_ignores_the_right_image() {
        let (left, right) = (image(8, 6), image(4, 4));
        assert!(validate(1.0, &left, None, false).is_ok());
        assert!(validate(1.0, &left, Some(&right), false).is_ok());
    }

    #[test]
    fn stereo_input_needs_a_matching_right_image() {
        let (left, right, small) = (image(8, 6), image(8, 6), image(4, 4));
        assert!(validate(1.0, &left, Some(&right), true).is_ok());
        assert!(matches!(
            validate(1.0, &left, None, true),
            Err(ProcessError::StereoRequired)
        ));
        assert!(matches!(
            validate(1.0, &left, Some(&small), true),
            Err(ProcessError::StereoSizeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_empty_images() {
        let empty = image(0, 6);
        assert!(matches!(
            validate(1.0, &empty, None, false),
            Err(ProcessError::EmptyImage)
        ));
    }

    #[test]
    fn rejects_non_finite_timestamps() {
        let left = image(8, 6);
        for timestamp_sec in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                validate(timestamp_sec, &left, None, false),
                Err(ProcessError::NonFiniteTimestamp { .. })
            ));
        }
    }
}
