

# --- loading libraries -------------------------------------------------------
import pandas as pd
import os
import sys
# ------------------------------------------------------ loading libraries ----


# --- I/O ---------------------------------------------------------------------
file_a = sys.argv[1]
file_b = sys.argv[2]
# --------------------------------------------------------------------- I/O ---
print('file a: ', file_a)
print('file b: ', file_b)
print()


# --- main routine ------------------------------------------------------------
df_a = pd.read_pickle(file_a, compression = 'zip')
df_b = pd.read_pickle(file_b, compression = 'zip')
df = pd.concat([df_a, df_b], axis = 0)

# Saving combined file
df.to_pickle(file_a, compression = 'zip')

# Removing already combined file
os.remove(file_b)
# ------------------------------------------------------------ main routine ---
