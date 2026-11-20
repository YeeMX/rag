"""50 条劳动争议场景基准集（初稿，人工编写，条号已按四部法律全文核对）。

字段说明：
- question：用户口语化的复合问题（故意还原真实咨询的表达，区别于"法条考试题"）；
- relevant：gold 法条 id 列表（评测检索用，按相关性从高到低）；
- hard_negatives：语义相近但法律上不适用的干扰法条（用来验证零样本过滤的价值）；
- topic：场景标签，用于分类分析；
- note：这道题在测什么能力（多意图/术语错位/近似条款区分…）。

注意：这是"检索评测集"的种子，最终 JSONL 由 data/build_benchmark.py 生成并校验。
"""

from __future__ import annotations

# 便于阅读：把常用法条 id 起个别名
LCL = "lcl:"
LL = "ll:"
LDA = "lda:"
SIL = "sil:"

CASES: list[dict] = [
    # ---------------- 试用期 ----------------
    {
        "id": "Q01", "topic": "试用期",
        "question": "公司跟我约定试用期6个月，但劳动合同只签了1年，这样合法吗？",
        "relevant": [f"{LCL}第十九条", f"{LCL}第八十三条"],
        "hard_negatives": [f"{LCL}第二十一条", f"{LCL}第七十条"],
        "note": "同一法条内的期限规则 + 违法约定试用期的赔偿",
    },
    {
        "id": "Q02", "topic": "试用期",
        "question": "试用期工资只给正式工资的70%，公司说试用期都这样，合法吗？",
        "relevant": [f"{LCL}第二十条", f"{LL}第四十八条"],
        "hard_negatives": [f"{LCL}第十九条", f"{LCL}第八十五条"],
        "note": "试用期工资底线（不得低于本单位同岗位80%且不低于当地最低工资）",
    },
    {
        "id": "Q03", "topic": "试用期",
        "question": "我在试用期被公司辞退了，他们说试用期随时可以辞退不用理由，是吗？",
        "relevant": [f"{LCL}第二十一条", f"{LCL}第三十九条", f"{LCL}第四十条"],
        "hard_negatives": [f"{LCL}第二十五条", f"{LCL}第七十一条"],
        "note": "试用期解除必须符合法定情形，不能'随时'",
    },
    {
        "id": "Q04", "topic": "多意图",
        "question": "试用期6个月、工资打7折、还说试用期不缴社保，这三件事公司都违法吗？",
        "relevant": [f"{LCL}第十九条", f"{LCL}第二十条", f"{SIL}第五十八条", f"{LCL}第三十八条"],
        "hard_negatives": [f"{LCL}第四十六条", f"{LL}第七十二条"],
        "note": "三意图复合问题：期限、工资、社保；考分解",
    },
    # ---------------- 未签书面合同 ----------------
    {
        "id": "Q05", "topic": "未签合同",
        "question": "入职8个月了公司一直不签书面劳动合同，我能要到多少双倍工资？",
        "relevant": [f"{LCL}第八十二条", f"{LCL}第十条", f"{LCL}第十四条"],
        "hard_negatives": [f"{LCL}第八十五条", f"{LCL}第四十六条"],
        "note": "二倍工资差额（次月起至满一年前一日，最多11个月）+ 视为无固定期限",
    },
    {
        "id": "Q06", "topic": "未签合同",
        "question": "公司没签劳动合同，还把我辞退了，我可以同时要未签合同的双倍工资和违法解除赔偿金吗？",
        "relevant": [f"{LCL}第八十二条", f"{LCL}第八十七条", f"{LCL}第四十七条"],
        "hard_negatives": [f"{LCL}第四十六条", f"{LCL}第八十四条"],
        "note": "多意图：两项请求权可并行主张（法条竞合辨析）",
    },
    {
        "id": "Q07", "topic": "未签合同",
        "question": "不签劳动合同满一年后，是不是就自动变成无固定期限合同了？",
        "relevant": [f"{LCL}第十四条", f"{LCL}第八十二条"],
        "hard_negatives": [f"{LCL}第十三条", f"{LCL}第十九条"],
        "note": "视为无固定期限劳动合同的法定情形",
    },
    # ---------------- 解除与赔偿 ----------------
    {
        "id": "Q08", "topic": "违法解除",
        "question": "工作3年7个月，月薪12000，公司没有任何理由把我辞退，能赔多少？",
        "relevant": [f"{LCL}第八十七条", f"{LCL}第四十七条"],
        "hard_negatives": [f"{LCL}第四十六条", f"{LCL}第四十条"],
        "note": "违法解除赔偿金 2N = 8 个月工资（3年7个月按4年计，2×4=8）",
    },
    {
        "id": "Q09", "topic": "经济补偿",
        "question": "公司在合同到期后不跟我续签了，我能拿到经济补偿吗？",
        "relevant": [f"{LCL}第四十六条", f"{LCL}第四十四条", f"{LCL}第四十七条"],
        "hard_negatives": [f"{LCL}第八十七条", f"{LCL}第三十九条"],
        "note": "合同期满终止且非劳动者不同意续订，应付经济补偿 N 而非 2N",
    },
    {
        "id": "Q10", "topic": "经济补偿",
        "question": "我月薪5万，当地社平工资1万，工作15年，经济补偿是不是按15个月×5万算？",
        "relevant": [f"{LCL}第四十七条"],
        "hard_negatives": [f"{LCL}第八十七条", f"{LCL}第九十七条"],
        "note": "高薪封顶规则：按3倍社平工资、年限最高12年",
    },
    {
        "id": "Q11", "topic": "经济补偿",
        "question": "经济补偿的月工资是按基本工资算，还是把奖金补贴都算进去？",
        "relevant": [f"{LCL}第四十七条", f"{LL}第四十八条"],
        "hard_negatives": [f"{LCL}第三十条", f"{LCL}第八十五条"],
        "note": "月工资=解除前12个月平均应得工资，含奖金津贴补贴",
    },
    {
        "id": "Q12", "topic": "无过失性辞退",
        "question": "我生病请了3个月病假，公司说医疗期满不能上班就辞退我，需要提前30天通知吗？",
        "relevant": [f"{LCL}第四十条", f"{LCL}第四十六条", f"{LCL}第四十七条"],
        "hard_negatives": [f"{LCL}第三十九条", f"{LCL}第四十二条"],
        "note": "医疗期满不能从事原工作 → 提前30日通知或支付代通知金 + 经济补偿",
    },
    {
        "id": "Q13", "topic": "不能解除",
        "question": "我怀孕5个月了，公司说业绩不好要裁员，可以解除我的劳动合同吗？",
        "relevant": [f"{LCL}第四十二条", f"{LCL}第四十五条", f"{LCL}第四十八条"],
        "hard_negatives": [f"{LCL}第四十一条", f"{LCL}第四十条"],
        "note": "孕期/产期/哺乳期不得依第40、41条解除，合同期满应续延",
    },
    {
        "id": "Q14", "topic": "劳动者解除",
        "question": "公司拖欠我两个月工资，我可以直接走人并要求经济补偿吗？",
        "relevant": [f"{LCL}第三十八条", f"{LCL}第四十六条", f"{LL}第五十条"],
        "hard_negatives": [f"{LCL}第三十七条", f"{LCL}第九十条"],
        "note": "被迫解除（未及时足额支付劳动报酬）→ 经济补偿；区别于主动辞职",
    },
    {
        "id": "Q15", "topic": "劳动者解除",
        "question": "我提前30天书面通知公司辞职，公司不批，我能走吗？要赔违约金吗？",
        "relevant": [f"{LCL}第三十七条", f"{LCL}第二十五条", f"{LCL}第九十条"],
        "hard_negatives": [f"{LCL}第三十八条", f"{LCL}第二十二条"],
        "note": "预告解除权；违约金仅限服务期与竞业限制两种法定情形",
    },
    {
        "id": "Q16", "topic": "经济性裁员",
        "question": "公司说经营困难要裁20个人，需要走什么程序？被裁的人有补偿吗？",
        "relevant": [f"{LCL}第四十一条", f"{LCL}第四十六条", f"{LCL}第四十七条"],
        "hard_negatives": [f"{LCL}第四十条", f"{LCL}第四十二条"],
        "note": "经济性裁员的人数/程序要件（提前30日向工会或全体职工说明等）",
    },
    {
        "id": "Q17", "topic": "服务期",
        "question": "公司出钱送我去培训，签了5年服务期，我干了2年想走，要赔多少违约金？",
        "relevant": [f"{LCL}第二十二条", f"{LCL}第二十五条"],
        "hard_negatives": [f"{LCL}第二十三条", f"{LCL}第九十条"],
        "note": "服务期违约金不得超过培训费用，且按未履行部分比例分摊",
    },
    # ---------------- 竞业限制 ----------------
    {
        "id": "Q18", "topic": "竞业限制",
        "question": "我离职后公司一直没给竞业限制补偿金，我还需要遵守竞业限制吗？",
        "relevant": [f"{LCL}第二十三条", f"{LCL}第二十四条"],
        "hard_negatives": [f"{LCL}第二十二条", f"{LCL}第九十条"],
        "note": "未支付补偿金满3个月，劳动者可请求解除竞业限制约定",
    },
    {
        "id": "Q19", "topic": "竞业限制",
        "question": "我只是普通员工，公司也让我签竞业限制协议，这有效吗？",
        "relevant": [f"{LCL}第二十四条", f"{LCL}第二十三条"],
        "hard_negatives": [f"{LCL}第二十二条", f"{LCL}第二十五条"],
        "note": "竞业限制人员范围：高级管理/高级技术/其他负有保密义务的人员",
    },
    {
        "id": "Q20", "topic": "违约金",
        "question": "劳动合同里约定我提前离职要赔公司10万违约金，这种条款有效吗？",
        "relevant": [f"{LCL}第二十五条", f"{LCL}第二十二条", f"{LCL}第二十三条"],
        "hard_negatives": [f"{LCL}第三十七条", f"{LCL}第九十条"],
        "note": "违约金法定情形限定；其余约定无效",
    },
    # ---------------- 加班与工时 ----------------
    {
        "id": "Q21", "topic": "加班费",
        "question": "我平时加班、周末加班、国庆加班都有，加班费应该按几倍算？",
        "relevant": [f"{LL}第四十四条", f"{LL}第四十一条"],
        "hard_negatives": [f"{LL}第四十二条", f"{LCL}第三十一条"],
        "note": "1.5/2/3 倍分档；法定节假日与休息日的区分",
    },
    {
        "id": "Q22", "topic": "加班费",
        "question": "公司实行996，每天工作12小时，我可以主张哪些权利？",
        "relevant": [f"{LL}第三十六条", f"{LL}第四十一条", f"{LL}第四十四条", f"{LCL}第三十一条"],
        "hard_negatives": [f"{LL}第四十二条", f"{LL}第三十八条"],
        "note": "多意图：工时上限、加班时长上限、加班费、拒绝强迫加班",
    },
    {
        "id": "Q23", "topic": "加班费",
        "question": "公司说加班费已经包含在月薪里了，实际没另外发，这说法成立吗？",
        "relevant": [f"{LL}第四十四条", f"{LCL}第三十条", f"{LL}第五十条"],
        "hard_negatives": [f"{LCL}第八十五条", f"{LL}第四十八条"],
        "note": "包薪制抗辩；工资应按约定足额支付，加班另计",
    },
    # ---------------- 工资 ----------------
    {
        "id": "Q24", "topic": "欠薪",
        "question": "公司拖欠工资3个月，我去劳动监察投诉还是直接申请仲裁？",
        "relevant": [f"{LL}第五十条", f"{LDA}第二条", f"{LDA}第九条", f"{LDA}第二十七条"],
        "hard_negatives": [f"{LCL}第八十五条", f"{LL}第九十一条"],
        "note": "欠薪救济路径：劳动监察 vs 劳动仲裁；仲裁时效",
    },
    {
        "id": "Q25", "topic": "扣工资",
        "question": "我迟到两次，公司扣了我半个月工资，这合法吗？",
        "relevant": [f"{LL}第五十条", f"{LCL}第三十条", f"{LL}第四十八条"],
        "hard_negatives": [f"{LCL}第八十五条", f"{LCL}第三十九条"],
        "note": "克扣工资的边界；最低工资保障",
    },
    {
        "id": "Q26", "topic": "最低工资",
        "question": "公司给我的工资低于当地最低工资标准，我可以要求补差吗？",
        "relevant": [f"{LL}第四十八条", f"{LL}第九十一条", f"{LCL}第八十五条"],
        "hard_negatives": [f"{LL}第四十四条", f"{LCL}第二十条"],
        "note": "最低工资保障与补差、加付赔偿金",
    },
    # ---------------- 社保 / 工伤 ----------------
    {
        "id": "Q27", "topic": "社保",
        "question": "公司一直不给我缴社保，我可以要求补缴吗？还能以此为由辞职要补偿吗？",
        "relevant": [f"{SIL}第五十八条", f"{SIL}第六十三条", f"{LCL}第三十八条", f"{LCL}第四十六条"],
        "hard_negatives": [f"{SIL}第八十四条", f"{LCL}第八十五条"],
        "note": "多意图：补缴（行政途径）+ 被迫解除经济补偿",
    },
    {
        "id": "Q28", "topic": "社保",
        "question": "公司跟我们约定'自愿放弃社保，多发现金'，这种协议有效吗？",
        "relevant": [f"{SIL}第五十八条", f"{LCL}第二十六条", f"{LCL}第三十八条"],
        "hard_negatives": [f"{SIL}第八十六条", f"{LCL}第三十五条"],
        "note": "社保强制缴纳，约定放弃无效（违反法律强制性规定）",
    },
    {
        "id": "Q29", "topic": "工伤",
        "question": "我在上班路上被车撞了，算工伤吗？公司没买工伤保险谁赔？",
        "relevant": [f"{LL}第十四条", f"{SIL}第四十一条", f"{SIL}第三十八条"],
        "hard_negatives": [f"{LL}第十三条", f"{SIL}第三十九条"],
        "note": "上下班途中非本人主要责任的交通事故属工伤；未参保由单位支付待遇",
    },
    {
        "id": "Q30", "topic": "工伤",
        "question": "工伤认定后我可以享受哪些待遇？停工留薪期工资怎么发？",
        "relevant": [f"{SIL}第三十三条", f"{SIL}第三十八条", f"{LL}第二十九条"],
        "hard_negatives": [f"{SIL}第四十五条", f"{LL}第二十八条"],
        "note": "停工留薪期待遇与工伤医疗待遇",
    },
    # ---------------- 医疗期 / 年休假 ----------------
    {
        "id": "Q31", "topic": "年休假",
        "question": "工作满2年，公司一天年假都不给，我可以要多少天？未休年假怎么补偿？",
        "relevant": [f"{LL}第四十五条", f"{LL}第四十四条"],
        "hard_negatives": [f"{LL}第三十八条", f"{LL}第四十一条"],
        "note": "年休假天数（1-10年5天）+ 未休年休假300%工资报酬",
    },
    {
        "id": "Q32", "topic": "医疗期",
        "question": "我工作4年、在本单位2年，医疗期最长可以休多久？",
        "relevant": [f"{LL}第二十六条", f"{LCL}第四十条"],
        "hard_negatives": [f"{LL}第二十九条", f"{LCL}第四十二条"],
        "note": "医疗期计算规则（实际工作年限/本单位工作年限）",
    },
    # ---------------- 调岗调薪 ----------------
    {
        "id": "Q33", "topic": "调岗降薪",
        "question": "公司没跟我商量就把我调到外地分公司，还降了30%工资，我可以拒绝吗？",
        "relevant": [f"{LCL}第三十五条", f"{LCL}第二十六条", f"{LCL}第三十八条"],
        "hard_negatives": [f"{LCL}第四十条", f"{LCL}第三十九条"],
        "note": "变更劳动合同需协商一致并书面；单方调岗降薪的风险",
    },
    {
        "id": "Q34", "topic": "劳动者权益",
        "question": "公司扣押了我的身份证和职业资格证书，说离职才还，怎么办？",
        "relevant": [f"{LCL}第九条", f"{LCL}第八十四条"],
        "hard_negatives": [f"{LCL}第二十二条", f"{LCL}第二十三条"],
        "note": "禁止扣押证件、收取财物；行政责任",
    },
    # ---------------- 仲裁程序 ----------------
    {
        "id": "Q35", "topic": "仲裁程序",
        "question": "公司欠我工资，我要先去劳动仲裁才能去法院起诉吗？仲裁要收费吗？",
        "relevant": [f"{LDA}第二条", f"{LDA}第五条", f"{LDA}第五十三条"],
        "hard_negatives": [f"{LDA}第二十七条", f"{LDA}第四十七条"],
        "note": "仲裁前置、仲裁免费",
    },
    {
        "id": "Q36", "topic": "仲裁时效",
        "question": "我离职已经一年半了才发现公司少发了加班费，现在申请仲裁是不是过期了？",
        "relevant": [f"{LDA}第二十七条"],
        "hard_negatives": [f"{LDA}第二十六条", f"{LL}第八十二条"],
        "note": "仲裁时效1年；劳动关系存续期间欠薪不受1年限制，终止后1年内提出",
    },
    {
        "id": "Q37", "topic": "仲裁程序",
        "question": "对仲裁裁决不服，我可以直接向法院起诉吗？多久内要起诉？",
        "relevant": [f"{LDA}第五十条", f"{LDA}第四十七条", f"{LDA}第四十八条"],
        "hard_negatives": [f"{LDA}第四十九条", f"{LL}第八十三条"],
        "note": "一裁终局 / 非终局裁决的起诉期限（15日）",
    },
    {
        "id": "Q38", "topic": "仲裁程序",
        "question": "公司不履行仲裁裁决，我该怎么强制执行？",
        "relevant": [f"{LDA}第五十一条", f"{LDA}第四十九条"],
        "hard_negatives": [f"{LDA}第五十条", f"{LDA}第四条"],
        "note": "裁决履行与强制执行",
    },
    # ---------------- 劳务派遣 / 非全日制 ----------------
    {
        "id": "Q39", "topic": "劳务派遣",
        "question": "我是劳务派遣工，被派到工厂干了3年，现在用工单位要退回我，我能要补偿吗？",
        "relevant": [f"{LCL}第五十八条", f"{LCL}第六十五条", f"{LCL}第四十六条"],
        "hard_negatives": [f"{LCL}第六十六条", f"{LCL}第七十一条"],
        "note": "派遣单位是用人单位；退回情形与经济补偿",
    },
    {
        "id": "Q40", "topic": "非全日制",
        "question": "我每天只干4小时，公司按小时结算工资，可以约定试用期吗？",
        "relevant": [f"{LCL}第七十条", f"{LCL}第六十八条", f"{LCL}第七十二条"],
        "hard_negatives": [f"{LCL}第十九条", f"{LCL}第七十一条"],
        "note": "非全日制用工不得约定试用期；小时计酬与结算周期",
    },
    {
        "id": "Q41",
        "topic": "非全日制",
        "question": "非全日制用工，公司可以随时开除我吗？需要给经济补偿吗？",
        "relevant": [f"{LCL}第七十一条"],
        "hard_negatives": [f"{LCL}第四十六条", f"{LCL}第八十七条"],
        "note": "非全日制双方均可随时终止，不支付经济补偿",
    },
    # ---------------- 规章制度 / 工会 ----------------
    {
        "id": "Q42", "topic": "规章制度",
        "question": "公司以我违反员工手册为由开除我，但员工手册我从没见过，这样合法吗？",
        "relevant": [f"{LCL}第四条", f"{LCL}第三十九条"],
        "hard_negatives": [f"{LCL}第四十条", f"{LCL}第八十条"],
        "note": "规章制度的民主程序与公示告知义务；未经公示不能作为解除依据",
    },
    {
        "id": "Q43",
        "topic": "规章制度",
        "question": "公司规定女员工入职3年内不能怀孕，这种规定有效吗？",
        "relevant": [f"{LCL}第四条", f"{LCL}第二十六条", f"{LCL}第四十二条"],
        "hard_negatives": [f"{LCL}第三十九条", f"{LL}第十三条"],
        "note": "规章制度不得违反法律；性别歧视条款无效",
    },
    {
        "id": "Q44",
        "topic": "工会",
        "question": "公司单方解除我的劳动合同，没有通知工会，这会影响解除效力吗？",
        "relevant": [f"{LCL}第四十三条", f"{LCL}第四十八条"],
        "hard_negatives": [f"{LCL}第三十九条", f"{LCL}第四十一条"],
        "note": "解除应事先通知工会；程序违法可能构成违法解除",
    },
    # ---------------- 特殊主体 / 其他 ----------------
    {
        "id": "Q45",
        "topic": "特殊主体",
        "question": "我在公司连续工作满10年了，可以要求签无固定期限劳动合同吗？",
        "relevant": [f"{LCL}第十四条", f"{LCL}第八十二条"],
        "hard_negatives": [f"{LCL}第十三条", f"{LCL}第十九条"],
        "note": "无固定期限劳动合同的法定情形（连续满10年/连续两次固定期限）",
    },
    {
        "id": "Q46",
        "topic": "退休返聘",
        "question": "我已经退休开始领养老金了，又被公司返聘，还算劳动关系吗？被辞退有补偿吗？",
        "relevant": [f"{LCL}第四十四条", f"{LCL}第二条"],
        "hard_negatives": [f"{LCL}第四十六条", f"{LCL}第八十七条"],
        "note": "享受养老保险待遇后劳动关系终止；返聘通常按劳务关系处理",
    },
    {
        "id": "Q47",
        "topic": "竞争与保密",
        "question": "我离职后去了竞争对手公司，但没签竞业限制协议，公司能告我吗？",
        "relevant": [f"{LCL}第二十三条", f"{LCL}第二十四条", f"{LCL}第九十条"],
        "hard_negatives": [f"{LCL}第二十二条", f"{LCL}第二十五条"],
        "note": "无竞业限制约定则不受限制；保密义务与竞业限制的区别",
    },
    {
        "id": "Q48",
        "topic": "多意图",
        "question": "公司让我连续加班3个月、没给加班费、还克扣绩效，我辞职的话能拿经济补偿吗？",
        "relevant": [f"{LL}第四十一条", f"{LL}第四十四条", f"{LCL}第三十八条", f"{LCL}第四十六条"],
        "hard_negatives": [f"{LCL}第三十七条", f"{LCL}第八十五条"],
        "note": "三意图复合：超时加班、加班费、克扣工资后被迫解除",
    },
    {
        "id": "Q49",
        "topic": "经济补偿",
        "question": "我在同一家公司工作了12年，公司提出协商解除劳动合同，补偿应该怎么算？",
        "relevant": [f"{LCL}第三十六条", f"{LCL}第四十六条", f"{LCL}第四十七条"],
        "hard_negatives": [f"{LCL}第八十七条", f"{LCL}第三十九条"],
        "note": "协商解除（用人单位提出）→ 经济补偿 N=12个月",
    },
    {
        "id": "Q50",
        "topic": "综合",
        "question": "公司没签合同、没缴社保、还拖欠工资，我被迫辞职后能主张哪些赔偿？该怎么维权？",
        "relevant": [f"{LCL}第八十二条", f"{SIL}第五十八条", f"{LCL}第三十八条", f"{LCL}第四十六条", f"{LDA}第二十七条"],
        "hard_negatives": [f"{LCL}第八十七条", f"{LCL}第八十五条"],
        "note": "综合题：多项请求权 + 程序路径 + 时效，考多意图分解与召回覆盖",
    },
]


def validate() -> list[str]:
    """自检：条号必须存在于语料中，relevant 不能为空。返回问题列表。"""
    import json

    from lrs.corpus import load_articles

    valid_ids = {a.id for a in load_articles()}
    problems: list[str] = []
    seen: set[str] = set()
    for case in CASES:
        if case["id"] in seen:
            problems.append(f"{case['id']}: id 重复")
        seen.add(case["id"])
        for key in ("relevant", "hard_negatives"):
            for aid in case[key]:
                if aid not in valid_ids:
                    problems.append(f"{case['id']}: {key} 中的 {aid} 不存在于语料")
        if not case["relevant"]:
            problems.append(f"{case['id']}: relevant 为空")
        for nid in case["hard_negatives"]:
            if nid in case["relevant"]:
                problems.append(f"{case['id']}: {nid} 同时出现在 relevant 与 hard_negatives")
    return problems
