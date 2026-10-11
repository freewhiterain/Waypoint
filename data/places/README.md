# 地点库

景点、餐厅、住宿的结构化数据，覆盖北京、上海、广州、深圳、成都、重庆、杭州、南京、苏州、武汉 10 个城市。

- `attractions.csv`：景点。类型、坐标、开门和关门时间、门票价格、建议游玩时长（小时）。
- `restaurants.csv`：餐厅。菜系、坐标、营业时间、人均价格、推荐菜。
- `accommodations.csv`：住宿。特色、坐标、价格、床位数。

`rank` 是源数据里的行号，源数据按热门程度排列，越小越靠前。

## 来源和许可

数据转换自 [ChinaTravel-Sandbox](https://huggingface.co/datasets/LAMDA-NeSy/ChinaTravel-Sandbox)（版本 2026.08.2），
许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。转换时只选取和重命名了列，没有修改数据内容。
转换脚本是 `scripts/import_chinatravel.py`。

```bibtex
@inproceedings{shao2026chinatravel,
  title     = {ChinaTravel: An Open-Ended Travel Planning Benchmark with Compositional Constraint Validation for Language Agents},
  author    = {Jie-Jing Shao and Bo-Wen Zhang and Xiao-Wen Yang and Baizhi Chen and Siyu Han and Pang Jinghao and Wen-Da Wei and Guohao Cai and Zhenhua Dong and Lan-Zhe Guo and Yu-Feng Li},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=0YRVlxY9BH}
}
```

这是静态快照，价格和营业时间可能与现状不符，行程里展示时要提醒用户以实际为准。
