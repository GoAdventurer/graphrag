# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

# 实体/关系抽取主 prompt。
# 这个 prompt 会被 GraphExtractor 填入：
# - {entity_types}: 允许抽取的实体类型列表，例如 organization/person/geo/event
# - {input_text}: 当前 text_unit 的文本内容
#
# 重要约束：
# - LLM 必须用 ("entity"<|>...) 输出实体
# - LLM 必须用 ("relationship"<|>...) 输出关系
# - 多条记录之间用 ## 分隔
# - 结束时输出 <|COMPLETE|>
# graph_extractor.py 会严格按这些分隔符解析模型输出，所以不要随意改格式。
GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
 
3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.
 
4. When finished, output <|COMPLETE|>
 
######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"<|>TECHGLOBAL<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
##
("entity"<|>VISION HOLDINGS<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal)
##
("relationship"<|>TECHGLOBAL<|>VISION HOLDINGS<|>Vision Holdings formerly owned TechGlobal from 2014 until present<|>5)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"<|>FIRUZABAD<|>GEO<|>Firuzabad held Aurelians as hostages)
##
("entity"<|>AURELIA<|>GEO<|>Country seeking to release hostages)
##
("entity"<|>QUINTARA<|>GEO<|>Country that negotiated a swap of money in exchange for hostages)
##
##
("entity"<|>TIRUZIA<|>GEO<|>Capital of Firuzabad where the Aurelians were being held)
##
("entity"<|>KROHAARA<|>GEO<|>Capital city in Quintara)
##
("entity"<|>CASHION<|>GEO<|>Capital city in Aurelia)
##
("entity"<|>SAMUEL NAMARA<|>PERSON<|>Aurelian who spent time in Tiruzia's Alhamia Prison)
##
("entity"<|>ALHAMIA PRISON<|>GEO<|>Prison in Tiruzia)
##
("entity"<|>DURKE BATAGLANI<|>PERSON<|>Aurelian journalist who was held hostage)
##
("entity"<|>MEGGIE TAZBAH<|>PERSON<|>Bratinas national and environmentalist who was held hostage)
##
("relationship"<|>FIRUZABAD<|>AURELIA<|>Firuzabad negotiated a hostage exchange with Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>AURELIA<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>FIRUZABAD<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>ALHAMIA PRISON<|>Samuel Namara was a prisoner at Alhamia prison<|>8)
##
("relationship"<|>SAMUEL NAMARA<|>MEGGIE TAZBAH<|>Samuel Namara and Meggie Tazbah were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>DURKE BATAGLANI<|>Samuel Namara and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>DURKE BATAGLANI<|>Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>FIRUZABAD<|>Samuel Namara was a hostage in Firuzabad<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>FIRUZABAD<|>Meggie Tazbah was a hostage in Firuzabad<|>2)
##
("relationship"<|>DURKE BATAGLANI<|>FIRUZABAD<|>Durke Bataglani was a hostage in Firuzabad<|>2)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

# 补抽 prompt。
# 第一轮抽取后，如果 max_gleanings > 0，GraphExtractor 会追加这条消息，
# 要求 LLM 在已有结果基础上继续补充遗漏的实体和关系，并保持相同输出格式。
CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
# 是否继续补抽的判断 prompt。
# LLM 只需要回答 Y 或 N：
# - Y: 仍有遗漏，继续补抽
# - N 或其他: 停止补抽
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"


# 中文实体/关系抽取 prompt 模板。
# 这个常量不会自动覆盖默认 prompt；如果输入资料主要是中文，可以把它写入
# prompts/extract_graph.txt，或在配置中把 extract_graph.prompt 指向自定义文件。
# 保留英文默认模板是为了兼容原项目示例和英文数据集。
GRAPH_EXTRACTION_PROMPT_ZH = """
-目标-
给定一段可能与当前任务相关的文本，以及允许抽取的实体类型列表，请识别文本中的实体，以及实体之间明确存在的关系。

-步骤-
1. 识别实体。每个实体必须包含以下信息：
- entity_name: 实体名称。英文实体请使用大写；中文实体保持原文名称，不要翻译。
- entity_type: 必须是以下类型之一：[{entity_types}]
- entity_description: 用简洁但完整的中文说明实体的属性、身份、行为或上下文。
实体输出格式必须为：("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. 在已识别实体中，找出明确相关的实体对。不要凭空扩展文本中没有依据的关系。
每条关系必须包含以下信息：
- source_entity: 第 1 步中识别到的源实体名称
- target_entity: 第 1 步中识别到的目标实体名称
- relationship_description: 用中文说明两个实体为什么相关，必须基于原文证据
- relationship_strength: 1 到 10 的数字，表示关系强度；直接、核心、反复出现的关系分数更高
关系输出格式必须为：("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. 只输出实体和关系列表。多条记录之间使用 **##** 分隔。

4. 完成后输出 <|COMPLETE|>

######################
-示例-
######################
Entity_types: 组织,人物,地点,事件
Text:
张三在北京参加了星河科技举办的人工智能大会，并代表公司介绍了新的知识图谱产品。
######################
Output:
("entity"<|>张三<|>人物<|>张三代表公司参加人工智能大会并介绍知识图谱产品)
##
("entity"<|>北京<|>地点<|>北京是人工智能大会的举办地点)
##
("entity"<|>星河科技<|>组织<|>星河科技是人工智能大会的举办方)
##
("entity"<|>人工智能大会<|>事件<|>人工智能大会是由星河科技举办的活动)
##
("relationship"<|>张三<|>人工智能大会<|>张三参加人工智能大会并介绍产品<|>8)
##
("relationship"<|>星河科技<|>人工智能大会<|>星河科技举办人工智能大会<|>9)
##
("relationship"<|>北京<|>人工智能大会<|>北京是人工智能大会的举办地点<|>6)
<|COMPLETE|>

######################
-真实数据-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""


CONTINUE_PROMPT_ZH = "上一次抽取仍然可能遗漏了实体或关系。请只补充符合前面实体类型的遗漏项，并严格使用相同输出格式：\n"

LOOP_PROMPT_ZH = "请判断是否仍有遗漏的实体或关系需要补充。如果有，请只回答 Y；如果没有，请只回答 N。\n"
