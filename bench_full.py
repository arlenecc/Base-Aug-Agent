"""Full index run with peak-memory and time monitoring."""
import os
import resource
import sys
import time

sys.path.insert(0, "src")

from agent.skill_retriever import SkillRetriever, SkillScanner

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024 / 1024

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (peak RSS {rss_gb():.2f} GB)", flush=True)

log("开始扫描 skills 目录")
t = time.time()
scanner = SkillScanner("/Users/arlenecc/base-agent-workspace/skills")
skills = scanner.scan(force=True)
log(f"扫描完成: {len(skills)} 个 skill，耗时 {time.time()-t:.1f}s")

tmp = "/tmp/skill_full_index"
os.makedirs(tmp, exist_ok=True)

retriever = SkillRetriever(
    skills_dir="/Users/arlenecc/base-agent-workspace/skills",
    db_path=os.path.join(tmp, "skills.lancedb"),
)

log(f"开始全量索引 {len(skills)} 个 skill (batch_size=8)")
t = time.time()
retriever._store.build(skills, batch_size=8)
log(f"索引完成: 耗时 {time.time()-t:.1f}s")

# 检索验证
t = time.time()
result = retriever.retrieve("analyze single cell RNA sequencing data", llm=None)
log(f"检索完成: 耗时 {time.time()-t:.2f}s")
print("候选:", flush=True)
for c in result["candidates"][:5]:
    print(f"  {c['dir']} | score={round(c['score'],3)}", flush=True)

retriever.close()
log("全部完成")
