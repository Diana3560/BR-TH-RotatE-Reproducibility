# 280 条分层样本结构一致性审计

从 R14 的 14 类关系中固定抽取 280 条三元组，每类关系 20 条，用于检查公开匿名数据与冻结结构约束的一致性。

Sampling design:
20 triples were sampled for each of the 14 R14 relations:
16 from Train, 2 from Validation, and 2 from Test.

审计项目包括：

- 样本三元组是否存在于对应的冻结 Train/Validation/Test 文件；
- 头实体与尾实体是否属于公开候选实体集合；
- 头尾实体类型是否与 Train-only 关系-类型 Schema 兼容。

结果：280/280 条样本通过上述结构检查。

“内部来源上下文可用性”仅作为补充信息记录，不作为本次结构一致性审计的通过条件
