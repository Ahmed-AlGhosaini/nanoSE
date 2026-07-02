from models.crn import CRNTiny

# Log folder identifier name
name = "crntiny_default"

# Training settings
epochs = 25
batch_size = 32
lr = 1e-3
wd = 0.02
compile = False
seed = 42
remix = False

# Model definition
model = CRNTiny()
