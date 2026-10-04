"""Copy a DAQmx TDMS prefix (default: 60 seconds) without rewriting its data."""

import argparse
import math
from pathlib import Path
import struct

from nptdms import TdmsFile


def trim_first60(
    source: Path, destination: Path, seconds: float = 60.0
) -> tuple[int, int]:
    """Copy whole segments, preserving all original metadata and raw bytes.

    Requires synchronous, fixed-width DAQmx channels and a segment boundary at
    the requested duration. Refuses other layouts rather than changing their
    representation.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be finite and positive.")
    metadata = TdmsFile.read_metadata(source, raw_timestamps=True)
    channels = [c for g in metadata.groups() for c in g.channels()]
    if not channels:
        raise ValueError("The source has no channels.")

    increment = float(channels[0].properties["wf_increment"])
    if not math.isfinite(increment) or increment <= 0:
        raise ValueError("wf_increment must be finite and positive.")
    samples_float = seconds / increment
    samples = round(samples_float)
    if not math.isclose(samples_float, samples, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{seconds:g} seconds does not correspond to a whole sample count.")

    bytes_per_sample = 0
    for channel in channels:
        if float(channel.properties["wf_increment"]) != increment:
            raise ValueError("All channels must have the same sampling interval.")
        if len(channel) != len(channels[0]) or len(channel) < samples:
            raise ValueError(
                f"Channels must have equal lengths and at least {seconds:g} seconds."
            )
        if channel.properties.get("wf_start_offset", 0.0) != channels[0].properties.get(
            "wf_start_offset", 0.0
        ):
            raise ValueError("All channels must have the same time offset.")
        raw_types = channel.scaler_data_types
        if len(raw_types) != 1:
            raise ValueError("Expected one fixed-width DAQmx scaler per channel.")
        raw_type = next(iter(raw_types.values()))
        if raw_type.size is None or raw_type.size <= 0:
            raise ValueError("Variable-width channel data is not supported.")
        bytes_per_sample += raw_type.size

    target_bytes = samples * bytes_per_sample
    raw_bytes = 0
    file_size = source.stat().st_size
    with source.open("rb") as reader:
        while raw_bytes < target_bytes:
            position = reader.tell()
            header = reader.read(28)
            if len(header) != 28:
                raise ValueError("Source ended before the requested segment boundary.")
            tag, flags, _, next_offset, data_offset = struct.unpack("<4sIIQQ", header)
            end = position + 28 + next_offset
            if tag != b"TDSm" or not data_offset <= next_offset or end > file_size:
                raise ValueError("Invalid or incomplete TDMS segment.")
            if not flags & (1 << 7) or not flags & (1 << 3):
                raise ValueError("Expected DAQmx raw-data segments.")
            raw_bytes += next_offset - data_offset
            reader.seek(end)
        if raw_bytes != target_bytes:
            raise ValueError(
                f"{seconds:g} seconds falls inside a segment; byte-preserving trim refused."
            )
        cutoff = reader.tell()
        metadata_segments = []
        while reader.tell() < file_size:
            position = reader.tell()
            header = reader.read(28)
            if len(header) != 28:
                raise ValueError("Incomplete trailing TDMS header.")
            tag, _, _, next_offset, data_offset = struct.unpack("<4sIIQQ", header)
            end = position + 28 + next_offset
            if tag != b"TDSm" or not data_offset <= next_offset or end > file_size:
                raise ValueError("Invalid or incomplete trailing TDMS segment.")
            # DAQmx files can store waveform timing and units only at file close.
            if next_offset == data_offset:
                metadata_segments.append((position, end - position))
            reader.seek(end)
        reader.seek(0)
        # Exclusive creation prevents overwriting the source or an existing result.
        with destination.open("xb") as writer:
            remaining = cutoff
            while remaining:
                block = reader.read(min(1024 * 1024, remaining))
                if not block:
                    raise EOFError("Source changed during copying.")
                writer.write(block)
                remaining -= len(block)
            for position, size in metadata_segments:
                reader.seek(position)
                block = reader.read(size)
                if len(block) != size:
                    raise EOFError("Source changed during metadata copying.")
                writer.write(block)

    result = TdmsFile.read_metadata(destination)
    expected = TdmsFile.read_metadata(source)
    result_channels = [c for g in result.groups() for c in g.channels()]
    if len(result_channels) != len(channels) or any(
        len(c) != samples for c in result_channels
    ):
        raise RuntimeError(f"Output sample-count verification failed: {destination}")
    expected_objects = [expected, *expected.groups()]
    result_objects = [result, *result.groups()]
    expected_objects.extend(c for g in expected.groups() for c in g.channels())
    result_objects.extend(result_channels)
    if len(expected_objects) != len(result_objects) or any(
        a.properties != b.properties
        for a, b in zip(expected_objects, result_objects)
    ):
        raise RuntimeError(f"Output metadata verification failed: {destination}")
    return samples, destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "data" / "Patricia" / "20250613" / "files3.tdms",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    destination = args.output or args.source.with_name(
        f"{args.source.stem}_first{args.seconds:g}s.tdms"
    )
    samples, size = trim_first60(args.source, destination, args.seconds)
    print(
        f"Created {destination}: {samples:,} samples/channel, "
        f"{size:,} bytes ({args.seconds:g} s)."
    )


if __name__ == "__main__":
    main()
