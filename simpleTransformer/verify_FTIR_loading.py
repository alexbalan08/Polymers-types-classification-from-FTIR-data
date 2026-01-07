from src.data.ftir_dataset import FTIRDataset

ftir = FTIRDataset("data/merged_postprocessed_FTIR.csv")
ftir.load()

X = ftir.get_spectra()
plastics = ftir.get_plastics()

print("FTIR shape:", X.shape)
print("Number of plastics:", len(plastics))
print("First plastic:", plastics[0])
print("First spectrum (first 10 values):", X[0][:10])

