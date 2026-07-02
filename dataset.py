import os
import random
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class VoiceBankDemandDataset(Dataset):
    def __init__(self, split="train", sample_rate=16000, duration=2.0, mock_size=None):
        self.sample_rate = sample_rate
        if split == "test":
            self.target_length = None
        else:
            self.target_length = (
                int(sample_rate * duration) if duration is not None else None
            )

        if split == "train":
            # Cache file configuration
            cache_suffix = f"_mock_{mock_size}" if mock_size is not None else ""
            self.cache_path = f"train_cache{cache_suffix}.bin"
            
            if os.path.exists(self.cache_path):
                print(f"Loading cached training dataset from {self.cache_path}...")
                with open(self.cache_path, "rb") as f:
                    bytes_data = f.read()
                flat_array = np.frombuffer(bytes_data, dtype=np.int16)
                
                # Reshape as (-1, fixed_length, 2)
                total_samples = len(flat_array) // 2
                num_segments = total_samples // self.target_length
                flat_array = flat_array[:num_segments * self.target_length * 2]
                self.cached_array = flat_array.reshape(-1, self.target_length, 2)
                print(f"Loaded {len(self.cached_array)} cached segments of length {self.target_length}.")
            else:
                print(f"Cache file {self.cache_path} not found. Loading split '{split}' of JacobLinCool/VoiceBank-DEMAND-16k...")
                full_ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split=split)
                if mock_size is not None:
                    full_ds = full_ds.select(range(min(mock_size, len(full_ds))))
                
                all_interleaved = []
                for sample in tqdm(full_ds, desc=f"Caching {split}"):
                    clean_wav = sample["clean"].get_all_samples().data.squeeze(0).numpy()
                    noisy_wav = sample["noisy"].get_all_samples().data.squeeze(0).numpy()
                    
                    clean_int16 = (clean_wav * 32767.0).astype(np.int16)
                    noisy_int16 = (noisy_wav * 32767.0).astype(np.int16)
                    
                    # Interleave clean and noisy: shape (L, 2)
                    interleaved = np.stack([noisy_int16, clean_int16], axis=-1)
                    all_interleaved.append(interleaved)
                
                # Concatenate all training examples together
                all_data = np.concatenate(all_interleaved, axis=0) # shape (Total_Samples, 2)
                
                print(f"Saving training dataset cache to {self.cache_path}...")
                with open(self.cache_path, "wb") as f:
                    f.write(all_data.tobytes())
                
                # Read from disk using np.frombuffer and reshape
                with open(self.cache_path, "rb") as f:
                    bytes_data = f.read()
                flat_array = np.frombuffer(bytes_data, dtype=np.int16)
                
                total_samples = len(flat_array) // 2
                num_segments = total_samples // self.target_length
                flat_array = flat_array[:num_segments * self.target_length * 2]
                self.cached_array = flat_array.reshape(-1, self.target_length, 2)
                print(f"Loaded {len(self.cached_array)} cached segments of length {self.target_length}.")
        else:
            # Custom binary caching for evaluation/test split
            cache_suffix = f"_mock_{mock_size}" if mock_size is not None else ""
            self.cache_path = f"test_cache{cache_suffix}.bin"
            
            self.clean_cached = []
            self.noisy_cached = []
            self.ids = []

            if os.path.exists(self.cache_path):
                print(f"Loading cached evaluation dataset from {self.cache_path}...")
                with open(self.cache_path, "rb") as f:
                    bytes_data = f.read()
                
                # Unpack custom header: number of samples (first 4 bytes as uint32)
                num_samples = np.frombuffer(bytes_data[:4], dtype=np.uint32)[0]
                sizes_offset = 4
                sizes_end = sizes_offset + 4 * num_samples
                
                # Extract sample lengths in frames (next 4 * num_samples bytes)
                sample_sizes = np.frombuffer(bytes_data[sizes_offset:sizes_end], dtype=np.uint32)
                
                # Remaining buffer represents audio bytes
                audio_offset = sizes_end
                flat_audio = np.frombuffer(bytes_data[audio_offset:], dtype=np.int16)
                
                curr_idx = 0
                for i, L in enumerate(sample_sizes):
                    end_idx = curr_idx + 2 * L
                    sample_data = flat_audio[curr_idx:end_idx].reshape(L, 2)
                    curr_idx = end_idx
                    
                    noisy_int16 = sample_data[:, 0]
                    clean_int16 = sample_data[:, 1]
                    
                    self.noisy_cached.append(torch.from_numpy(noisy_int16.copy()))
                    self.clean_cached.append(torch.from_numpy(clean_int16.copy()))
                    self.ids.append(f"test_{i}")
                print(f"Loaded {num_samples} cached evaluation samples.")
            else:
                print(f"Cache file {self.cache_path} not found. Loading split '{split}' of JacobLinCool/VoiceBank-DEMAND-16k...")
                full_ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split=split)
                if mock_size is not None:
                    full_ds = full_ds.select(range(min(mock_size, len(full_ds))))
                
                sample_sizes = []
                audio_data_list = []

                print(f"Caching dataset to memory and writing cache to {self.cache_path}...")
                for sample in tqdm(full_ds, desc=f"Caching {split}"):
                    clean_wav = sample["clean"].get_all_samples().data.squeeze(0).numpy()
                    noisy_wav = sample["noisy"].get_all_samples().data.squeeze(0).numpy()

                    clean_int16 = (clean_wav * 32767.0).astype(np.int16)
                    noisy_int16 = (noisy_wav * 32767.0).astype(np.int16)
                    
                    L = len(clean_int16)
                    sample_sizes.append(L)

                    # Interleave clean and noisy: shape (L, 2)
                    interleaved = np.stack([noisy_int16, clean_int16], axis=-1)
                    audio_data_list.append(interleaved)

                    self.clean_cached.append(torch.from_numpy(clean_int16))
                    self.noisy_cached.append(torch.from_numpy(noisy_int16))
                    self.ids.append(sample["id"])

                num_samples = len(sample_sizes)
                header = np.array([num_samples], dtype=np.uint32).tobytes()
                sizes_bytes = np.array(sample_sizes, dtype=np.uint32).tobytes()
                flat_audio = np.concatenate(audio_data_list, axis=0)
                audio_bytes = flat_audio.tobytes()

                full_buffer = header + sizes_bytes + audio_bytes
                with open(self.cache_path, "wb") as f:
                    f.write(full_buffer)
                print(f"Successfully cached {num_samples} samples to {self.cache_path}.")

    def __len__(self):
        if hasattr(self, "cached_array"):
            return len(self.cached_array)
        return len(self.clean_cached)

    def __getitem__(self, idx):
        if hasattr(self, "cached_array"):
            segment = self.cached_array[idx] # shape: (fixed_length, 2)
            noisy_int16 = segment[:, 0]
            clean_int16 = segment[:, 1]

            noisy_f32 = torch.from_numpy(noisy_int16).to(torch.float32) / 32767.0
            clean_f32 = torch.from_numpy(clean_int16).to(torch.float32) / 32767.0
            return noisy_f32, clean_f32

        # Eval loader path (restored to original sample lengths)
        noisy_int16 = self.noisy_cached[idx]
        clean_int16 = self.clean_cached[idx]

        noisy_f32 = noisy_int16.to(torch.float32) / 32767.0
        clean_f32 = clean_int16.to(torch.float32) / 32767.0

        if self.target_length is None:
            return noisy_f32, clean_f32

        num_samples = noisy_f32.size(0)

        if num_samples >= self.target_length:
            start = random.randint(0, num_samples - self.target_length)
            noisy_sliced = noisy_f32[start : start + self.target_length]
            clean_sliced = clean_f32[start : start + self.target_length]
        else:
            pad_len = self.target_length - num_samples
            noisy_sliced = torch.nn.functional.pad(noisy_f32, (0, pad_len))
            clean_sliced = torch.nn.functional.pad(clean_f32, (0, pad_len))

        return noisy_sliced, clean_sliced
