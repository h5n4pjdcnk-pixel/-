import os

# 强制绕过代理
os.environ['NO_PROXY'] = '127.0.0.1,localhost'

# 大模型配置
LLM_CONFIG = {

}

NEO4J_CONF = {
   
}

# PDF处理配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100