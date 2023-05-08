from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model import TabularModel
from sklearn.model_selection import train_test_split

target_name = ["type"]

cat_col_names = []

num_col_names = attrs.columns.tolist()
num_col_names.remove("type")

feature_columns = num_col_names + cat_col_names + target_name

df = attrs.copy()
train, test = train_test_split(df, random_state=42)
train, val = train_test_split(train, random_state=42)
num_classes = len(set(train[target_name].values.ravel()))

data_config = DataConfig(
    target=target_name,
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform=None,  # "quantile_normal",
    normalize_continuous_features=True,
)
head_config = LinearHeadConfig(layers="", dropout=0.1, initialization="kaiming").__dict__
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    metrics=["f1_score", "accuracy"],
    metrics_params=[{"num_classes": num_classes}, {}],
)
trainer_config = TrainerConfig(
    auto_lr_find=True, fast_dev_run=False, max_epochs=500, batch_size=512
)
optimizer_config = OptimizerConfig()
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(
    train=train,
    validation=val,
)

pred_df = tabular_model.predict(test)
tabular_model.save_model("examples/test_save")

conus = pynhd.streamcat(names, conus=True, metric_areas="watershed")
conus = conus.dropna(axis=1, how="all")
conus = conus.dropna(axis=0)
conus.columns = conus.columns.str.lower()
pred_df = tabular_model.predict(conus)
pred_df.head()
type_mapping_inv = {v: k for k, v in type_mapping.items()}
pred_df["type"] = pred_df["prediction"].map(type_mapping_inv)
cat = pynhd.nhdplus_l48("CatchmentSP")
drought_cat = pd.merge(cat, pred_df, left_on="FEATUREID", right_on="comid", how="right")
drought_cat.plot(column="type", figsize=(10, 10), legend=True)
