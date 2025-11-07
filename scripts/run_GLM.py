#!/usr/bin/env python3
import src2.differential_splicing as ds
import polars as pl
import pandas as pd
import argparse
from joblib import Parallel, delayed
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run GLM for differential splicing analysis.")
    parser.add_argument("--sharing", type=str, required=True, help="Grouping variable ('three' or 'five').")
    parser.add_argument("--mode", type=str, required=True, help="Sharing mode ('three' or 'five').")
    parser.add_argument("--model", type=str, required=True, help="Model type ('simple' or 'multiple').")
    parser.add_argument("--predictor", type=str, required=True, help="Predictor variable for the model.")
    parser.add_argument("--chunk", type=int, required=True, help="Chunk number.")   
    args = parser.parse_args()

    ratio_matrix = pl.read_parquet(f"proc/ratio_matrix_{args.sharing}_{args.mode}.parquet")
    ephys_data = pd.read_parquet(f"proc/ephys_data_{args.sharing}_{args.mode}.parquet")

    with open (f"proc/ratio_matrix/{args.sharing}_{args.mode}/{args.chunk}.txt", "r") as f:
        SJ_list = f.read().splitlines()
    
    if args.predictor == "cpm":
        #TODO: implement cpm regression joblib version
        for intron_group in SJ_list:
            gene_name = intron_group.split("_")[0]
            if args.model == "simple":
                reduced = "1"
                full = f"{gene_name}"
            elif args.model == "multiple":
                reduced = "subclass"
                full = f"subclass + {gene_name}"
            else:
                raise ValueError("Model type not recognized. Use 'simple' or 'multiple'.")
            results = ds.run_regression(ratio_matrix, ephys_data, intron_group, reduced, full)
    else:
        if args.model == "simple":
            reduced = "1"
            full = f"{args.predictor}"
        elif args.model == "multiple":
            reduced = "subclass"
            full = f"subclass + {args.predictor}"
        else:
            raise ValueError("Model type not recognized. Use 'simple' or 'multiple'.")
        results = [ds.run_regression(ratio_matrix, ephys_data, intron_group, reduced, full) for intron_group in SJ_list]
        if Path(f"proc/{args.sharing}_{args.mode}_{args.model}/{args.predictor}").exists() is False:
            print(f"Creating directory proc/{args.sharing}_{args.mode}_{args.model}/{args.predictor}")
            Path(f"proc/{args.sharing}_{args.mode}_{args.model}/{args.predictor}").mkdir(parents=True)
        with open (f"proc/{args.sharing}_{args.mode}_{args.model}/{args.predictor}/{args.chunk}.pkl", "wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main()

