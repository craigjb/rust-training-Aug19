use clap::{crate_version, App, Arg};
use image::{GrayImage, Luma};

/// Trait for grayscale images to get a pixel's luma value with the
/// edges "extended" past the boundary of the image.
trait ImageLumaExtended {
    fn get_pixel_luma_extended(&self, x: i32, y: i32) -> f32;
    fn put_pixel_luma_extended(&mut self, x: i32, y: i32, luma: f32);
}

impl ImageLumaExtended for GrayImage {
    fn get_pixel_luma_extended(&self, x: i32, y: i32) -> f32 {
        self.get_pixel(
            x.max(0).min(self.width() as i32 - 1) as u32,
            y.max(0).min(self.height() as i32 - 1) as u32,
        )[0] as f32
            / 255.0
    }

    fn put_pixel_luma_extended(&mut self, x: i32, y: i32, luma: f32) {
        self.put_pixel(
            x.max(0).min(self.width() as i32 - 1) as u32,
            y.max(0).min(self.height() as i32 - 1) as u32,
            Luma([(luma * 255.0) as u8]),
        );
    }
}

/// Kernel for the Sobel operator in the X direction
const SOBEL_KERNEL_X: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

/// Kernel for the Sobel operator in the Y direction
const SOBEL_KERNEL_Y: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

/// Convolves a given kernel with a block of pixels in normalize f32 format
fn convolve(kernel: &[[f32; 3]; 3], pixels: &[[f32; 3]; 3]) -> f32 {
    let accumulator: f32 = kernel
        .iter()
        .zip(pixels.iter())
        .flat_map(|(kernel_row, input_row)| {
            kernel_row
                .iter()
                .zip(input_row.iter())
                .map(|(kernel, input)| kernel * input)
        })
        .sum();
    // normalize
    accumulator / 8.0
}

/// Applies the Sobel operator to an entire grayscale image
fn sobel_filter(input: &GrayImage) -> GrayImage {
    let mut result = input.clone();

    for x in 0..(input.width() as i32) {
        for y in 0..(input.height() as i32) {
            let pixels = [
                [
                    input.get_pixel_luma_extended(x - 1, y - 1),
                    input.get_pixel_luma_extended(x - 1, y),
                    input.get_pixel_luma_extended(x - 1, y + 1),
                ],
                [
                    input.get_pixel_luma_extended(x, y - 1),
                    input.get_pixel_luma_extended(x, y),
                    input.get_pixel_luma_extended(x, y + 1),
                ],
                [
                    input.get_pixel_luma_extended(x + 1, y - 1),
                    input.get_pixel_luma_extended(x + 1, y),
                    input.get_pixel_luma_extended(x + 1, y + 1),
                ],
            ];

            let gradient_x = convolve(&SOBEL_KERNEL_X, &pixels);
            let gradient_y = convolve(&SOBEL_KERNEL_Y, &pixels);
            let magnitude = (gradient_x.powi(2) + gradient_y.powi(2)).sqrt();
            result.put_pixel_luma_extended(x, y, magnitude);
        }
    }

    result
}

fn main() {
    let matches = App::new("Sobel Filter")
        .version(crate_version!())
        .about("Applies a Sobel filter to an image")
        .arg(
            Arg::with_name("input")
                .required(true)
                .takes_value(true)
                .value_name("INPUT")
                .help("Input image file to filter"),
        )
        .arg(
            Arg::with_name("output")
                .required(true)
                .takes_value(true)
                .value_name("OUTPUT")
                .help("Output file for the filtered image"),
        )
        .get_matches();

    // The unwrap() on value_of is ok because clap ensures that argument is required.
    // If the argument were optional, we would need to handle the case wherein the
    // argument is not present (None).
    // The names passed to value_of(..) are the same names passed to
    // Arg::with_name(..) above.
    let input_path = matches.value_of("input").unwrap();
    let output_path = matches.value_of("output").unwrap();

    // image::open automatically detects the format from the file extension
    let input_image = image::open(input_path).expect("Failed to open input image file");

    // convert the input image to grayscale for processing
    let grayscale_image = input_image.to_luma();

    // apply the filter
    let output_image = sobel_filter(&grayscale_image);

    // The save function automatically determines the format, just like open
    output_image
        .save(output_path)
        .expect("Failed to save image to output file");
}
