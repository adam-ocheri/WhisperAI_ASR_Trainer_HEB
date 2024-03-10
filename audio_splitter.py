from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    """Normalize given audio chunk"""
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def split_audio_file(
    filepath,
    min_silence_len=2000,
    silence_thresh=-16,
    silence_duration=500,
    target_dBFS=-20,
):
    # Load your audio.
    audio_file = AudioSegment.from_mp3(filepath)
    base_file_duration = len(audio_file) * 0.001

    # Split track where the silence is 2 seconds or more and get chunks using
    # the imported function.
    chunks = split_on_silence(
        audio_file,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=True,
    )

    chunk_files = []
    accumulated_duration = 0

    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=silence_duration)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, target_dBFS)

        # Extract precise timestamp of this chunk, in relation to the base source file
        chunk_start_time = accumulated_duration
        accumulated_duration += len(chunk)

        new_filename = f".//{filepath}_chunk-{i}.wav"

        # Export the audio chunk with new bitrate.
        print("Exporting chunk{0}.mp3.".format(i))
        normalized_chunk.export(new_filename, format="wav")
        chunk_files.append({"file": new_filename, "time": chunk_start_time})
    return chunk_files


# This is the "ok values" for now
# split_audio_file("audio_files/otr4_chn0.mp3", 900, -40, 1200, -20)

# Here I test! will you send me an angel?
# split_audio_file("c0_service-person_otr4S", 300, -35, 0, -20)
