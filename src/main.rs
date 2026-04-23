use anyhow::{bail, Result};
use clap::Parser;
use std::path::PathBuf;

mod analysis;
mod audio;

#[derive(Parser)]
#[command(name = "audio-loop-finder")]
#[command(about = "Find repeating sections in an audio file using FFT-based normalized cross-correlation")]
struct Args {
    /// Input audio file (ogg / wav / mp3)
    file: PathBuf,

    /// Minimum time gap in seconds between matched timestamps
    #[arg(long)]
    minimum_length: Option<f64>,

    /// Minimum similarity threshold (0.0–1.0)
    #[arg(long)]
    minimum_similarity: Option<f64>,

    /// Number of results to show
    #[arg(long, default_value_t = 5)]
    limit: usize,

    /// Only show results where at least one timestamp falls in this range (e.g. 10.0-30.5).
    /// Can be specified multiple times.
    #[arg(long)]
    timestamp_range: Vec<String>,
}

fn parse_range(s: &str) -> Result<(f64, f64)> {
    let (lhs, rhs) = s
        .split_once('-')
        .ok_or_else(|| anyhow::anyhow!("expected START-END, got '{s}'"))?;
    let start: f64 = lhs
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid start value in '{s}'"))?;
    let end: f64 = rhs
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid end value in '{s}'"))?;
    if start > end {
        bail!("range start {start} must be ≤ end {end}");
    }
    Ok((start, end))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let ranges: Vec<(f64, f64)> = args
        .timestamp_range
        .iter()
        .map(|s| parse_range(s))
        .collect::<Result<_>>()?;

    let audio = audio::load(&args.file)?;

    let matches = analysis::find_matches(
        &audio.samples,
        audio.sample_rate,
        args.minimum_length,
        args.minimum_similarity,
        args.limit,
        &ranges,
    );

    if matches.is_empty() {
        println!("No matches found.");
    } else {
        for (i, m) in matches.iter().enumerate() {
            println!(
                "[{}] {:>9.3}s - {:>9.3}s  similarity: {:.3}",
                i + 1,
                m.time1,
                m.time2,
                m.similarity,
            );
        }
    }

    Ok(())
}
