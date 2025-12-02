"""
Query改写模块：解决词汇不匹配问题
将用户的自然问题改写为更匹配文档的形式
"""

# 课程特定的术语映射
COURSE_QUERY_REWRITES = {
    # 教师相关
    "instructor": ["taught by", "prof", "professor", "AP"],
    "teacher": ["taught by", "prof", "professor", "AP"],
    "professor": ["prof", "taught by", "AP"],
    
    # 课程相关
    "class": ["course", "section"],
    "homework": ["assignment", "practice"],
    "exam": ["test", "evaluation"],
}

def rewrite_query(query: str) -> list[str]:
    """
    将查询改写为多个可能的变体
    
    Args:
        query: 原始查询
        
    Returns:
        包含原始查询和改写变体的列表
    """
    queries = [query]  # 保留原始查询
    
    query_lower = query.lower()
    
    # 检查是否包含需要改写的词
    for key_term, alternatives in COURSE_QUERY_REWRITES.items():
        if key_term in query_lower:
            # 为每个替代词生成新查询
            for alt in alternatives:
                new_query = query_lower.replace(key_term, alt)
                queries.append(new_query)
    
    # 特殊的问答对改写
    if "who is" in query_lower and "instructor" in query_lower:
        queries.extend([
            "Chengwei Qin",
            "taught by Prof",
            "course taught by",
            "AP@HKUST"
        ])
    
    return queries


def expand_query_with_synonyms(query: str) -> str:
    """
    在原查询中添加同义词，用OR连接
    
    Args:
        query: 原始查询
        
    Returns:
        扩展后的查询
    """
    terms = query.lower().split()
    expanded_terms = []
    
    for term in terms:
        # 添加原词
        expanded_terms.append(term)
        
        # 添加同义词
        if term in COURSE_QUERY_REWRITES:
            expanded_terms.extend(COURSE_QUERY_REWRITES[term])
    
    return " ".join(set(expanded_terms))


if __name__ == "__main__":
    # 测试
    test_queries = [
        "who is the instructor",
        "what is the homework",
        "when is the exam"
    ]
    
    for q in test_queries:
        print(f"\n原始查询: {q}")
        rewrites = rewrite_query(q)
        print(f"改写变体 ({len(rewrites)}):")
        for i, r in enumerate(rewrites, 1):
            print(f"  {i}. {r}")
        
        expanded = expand_query_with_synonyms(q)
        print(f"扩展查询: {expanded}")
