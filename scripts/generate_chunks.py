#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path

def split_dataframe_into_chunks(ratio_matrix_path: str, sharing: str, mode: str, num_chunks: int = 60):
    """
    Given a polars dataframe where the first column is an index column, give me a function that splits the dataframe into n chunks
    where each chunk has (number_of_columns - 1)/n columns (excluding the index column), and that each column contains the index column.
    If the number of columns - 1 is not divisible by n, then the last chunk should contain the remaining columns.
    """
    if not Path(f"proc/ratio_matrix").exists():
        Path(f"proc/ratio_matrix").mkdir(parents=True, exist_ok=True)
    else:
        print(f"Directory proc/ratio_matrix already exists.")
    
    ratio_matrix = pl.read_parquet(f"proc/ratio_matrix_{sharing}_{mode}.parquet")
    SJ_list = ratio_matrix.columns[1:]
    SJ_list = list(set([s.rsplit('_', 1)[0] for s in SJ_list]))
    
    chunk_size = len(SJ_list) // num_chunks
    for i in range(1, num_chunks + 1):
        if i == num_chunks:
            end = chunk_size * num_chunks + (len(SJ_list) % num_chunks)
            my_list = SJ_list[chunk_size * (num_chunks - 1):end]
            with open (f"proc/ratio_matrix/{sharing}_{mode}/{i}.txt", "w") as f:
                f.writelines("\n".join(my_list))
        else:
            end = chunk_size * i
            my_list = SJ_list[end - chunk_size:end]
            with open (f"proc/ratio_matrix/{sharing}_{mode}/{i}.txt", "w") as f:
                f.writelines("\n".join(my_list))

def main():
    parser = argparse.ArgumentParser(description="Generate chunks of the ratio matrix and ephys data.")
    parser.add_argument("--num_chunks", type=int, default=60, help="Number of chunks to split the data into.")
    parser.add_argument("--sharing", type=str, default="none", help="Sharing strategy for the chunks.")
    parser.add_argument("--mode", type=str, help="cell or ttype")

    args = parser.parse_args()

    split_dataframe_into_chunks(f"proc/ratio_matrix_{args.sharing}_{args.mode}.parquet", args.sharing, args.mode, args.num_chunks)

if __name__ == "__main__":
    main()