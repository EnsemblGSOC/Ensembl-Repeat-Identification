# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pathlib

# %%
import matplotlib.pyplot as plt

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from utils import hits_column_dtypes, hits_to_dataframe

# %%
figsize = (16, 9)

# %%
data_directory = pathlib.Path("../data")

annotations_path = data_directory / "annotations" / "hg38.hits"

# %%
data = hits_to_dataframe(annotations_path)

# %%
data.head()

# %%
# show random sample of items
num_items = 10
random_seed = 5
data.sample(num_items, random_state=random_seed).sort_index()

# %%
# concise summary of dataframe
data.info()

# %%
# number of distinct elements
data.nunique()

# %% [markdown]
# ## column value counts

# %%
categorical_variables = {
    "seq_name",
    "family_acc",
    "family_name",
    "strand",
}

# %%
for column in categorical_variables:
    column_value_counts = data[column].value_counts()
    print(f"{column=}\n", column_value_counts, "\n")
    column_value_counts[:30]
    column_value_counts[:30].plot(kind="bar", figsize=figsize)
    plt.show()

# %% [markdown] tags=[]
# ## repeats length

# %%
data["ali_length"] = abs(data["ali-en"] - data["ali-st"]) + 1

# %%
data.head()

# %%
data["ali_length"].value_counts()

# %%
data.loc[data["ali_length"].sort_values().index][-10:]

# %%
ali_length_mean = data["ali_length"].mean()
ali_length_median = data["ali_length"].median()
ali_length_standard_deviation = data["ali_length"].std()

print(
    f"ali_length mean: {ali_length_mean:.2f}, median: {ali_length_median:.2f}, standard deviation: {ali_length_standard_deviation:.2f}"
)

# %%
figure = plt.figure()
ax = data["ali_length"].hist(figsize=figsize, bins=256)
ax.axvline(x=ali_length_mean, color="black", linewidth=1)
for count in range(1, 3 + 1):
    x = round(ali_length_mean + count * ali_length_standard_deviation)
    ax.axvline(x=x, color="red", linewidth=1)
ax.set(xlabel="ali_length", ylabel="number of repeats")
figure.add_axes(ax)

# %%
