use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

const FFT_SIZE: usize = 4096;
const N_BANDS: usize = 64;
const MAX_FRAMES: usize = 2000;

pub struct LoopMatch {
    pub time1: f64,
    pub time2: f64,
    pub similarity: f64,
}

pub fn find_matches(
    samples: &[f32],
    sample_rate: u32,
    minimum_length: Option<f64>,
    minimum_similarity: Option<f64>,
    limit: usize,
    timestamp_ranges: &[(f64, f64)],
) -> Vec<LoopMatch> {
    let sr = sample_rate as f64;
    let hop_size = (FFT_SIZE / 2).max(samples.len() / MAX_FRAMES);

    let frames = compute_frames(samples, hop_size);
    let n_frames = frames.len();
    if n_frames < 2 {
        return vec![];
    }

    let min_frames = minimum_length
        .map(|ml| ((ml * sr / hop_size as f64).ceil() as usize).max(1))
        .unwrap_or(1);

    let min_sim = minimum_similarity.map(|v| v as f32).unwrap_or(f32::NEG_INFINITY);

    let mut candidates: Vec<(u32, u32, f32)> = Vec::new();
    for i in 0..n_frames {
        for j in (i + min_frames)..n_frames {
            let sim = dot(&frames[i], &frames[j]);
            if sim >= min_sim {
                candidates.push((i as u32, j as u32, sim));
            }
        }
    }

    candidates.sort_unstable_by(|a, b| {
        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut results = Vec::new();
    for (i, j, sim) in candidates {
        let t1 = i as f64 * hop_size as f64 / sr;
        let t2 = j as f64 * hop_size as f64 / sr;

        if !timestamp_ranges.is_empty() {
            let in_range = timestamp_ranges
                .iter()
                .any(|(s, e)| (t1 >= *s && t1 <= *e) || (t2 >= *s && t2 <= *e));
            if !in_range {
                continue;
            }
        }

        results.push(LoopMatch {
            time1: t1,
            time2: t2,
            similarity: sim as f64,
        });
        if results.len() >= limit {
            break;
        }
    }

    results
}

fn compute_frames(samples: &[f32], hop_size: usize) -> Vec<Vec<f32>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (FFT_SIZE - 1) as f32).cos()))
        .collect();

    let n_bins = FFT_SIZE / 2 + 1;
    let band_edges = log_band_edges(n_bins, N_BANDS);

    let n_frames = if samples.len() >= FFT_SIZE {
        (samples.len() - FFT_SIZE) / hop_size + 1
    } else {
        0
    };

    let mut frames = Vec::with_capacity(n_frames);
    let mut buffer = vec![Complex::new(0.0f32, 0.0); FFT_SIZE];

    for i in 0..n_frames {
        let start = i * hop_size;
        for k in 0..FFT_SIZE {
            let s = samples.get(start + k).copied().unwrap_or(0.0);
            buffer[k] = Complex::new(s * hann[k], 0.0);
        }
        fft.process(&mut buffer);

        let magnitude: Vec<f32> = buffer[..n_bins].iter().map(|c| c.norm()).collect();

        let mut bands = vec![0.0f32; N_BANDS];
        for (idx, &(lo, hi)) in band_edges.iter().enumerate() {
            let count = hi - lo;
            bands[idx] = magnitude[lo..hi].iter().sum::<f32>() / count as f32;
        }

        let norm: f32 = bands.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut bands {
                *x /= norm;
            }
        }

        frames.push(bands);
    }

    frames
}

fn log_band_edges(n_bins: usize, n_bands: usize) -> Vec<(usize, usize)> {
    let log_min = 1.0f32.ln();
    let log_max = (n_bins as f32).ln();

    (0..n_bands)
        .map(|i| {
            let lo = (log_min + (log_max - log_min) * i as f32 / n_bands as f32).exp() as usize;
            let hi =
                (log_min + (log_max - log_min) * (i + 1) as f32 / n_bands as f32).exp() as usize;
            let lo = lo.clamp(0, n_bins - 1);
            let hi = hi.clamp(lo + 1, n_bins);
            (lo, hi)
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
