# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

# 实体/关系描述总结 prompt。
# 输入来自前面实体/关系合并阶段：
# - {entity_name}: 实体 title，或关系的 (source, target)
# - {description_list}: 同一个实体/关系在多个 text_unit 中抽到的描述列表
# - {max_length}: 最终描述的最大词数
#
# 目标：
# 将多条重复、互补甚至有冲突的描述合并成一条完整、连贯、第三人称描述。
# 这个 prompt 只影响 description 字段，不改变实体名、关系两端或权重等结构字段。
SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
Limit the final description length to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""
