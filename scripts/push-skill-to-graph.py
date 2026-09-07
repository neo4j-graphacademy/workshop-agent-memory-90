#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "neo4j",
#   "pyyaml",
#   "python-dotenv",
# ]
# ///
"""Project a compiled AIP skill's execution graph into this workshop's
shared Neo4j instance, so it can be inspected in Browser.

AIP's own SKILL.md format already models a skill as a graph - steps as
nodes, depends_on/inputs/outputs as edges (see .claude/skills/aip/SKILL.md
Sec "Prioritize scripts/"). This script reads that structure straight out
of SKILL.md and writes it as real nodes and relationships: :Skill,
:Step (--> next via DEPENDS_ON, or NEXT for plain list order),
:AntiPattern, :Scenario. Nothing here re-validates the skill - run AIP's
own validate.py for that first.

Usage: uv run scripts/push-skill-to-graph.py <skill-name>
"""

import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def parse_skill(skill_dir: Path) -> tuple[dict, dict]:
    """Split SKILL.md into its frontmatter and its single fenced YAML body."""
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter_text, rest = text.split("---", 2)
    frontmatter = yaml.safe_load(frontmatter_text)
    body_match = re.search(r"```yaml\n(.*?)```", rest, re.DOTALL)
    if not body_match:
        raise ValueError("SKILL.md body must be exactly one fenced yaml block")
    body = yaml.safe_load(body_match.group(1))
    return frontmatter, body


def push(driver, database: str, name: str, frontmatter: dict, body: dict) -> None:
    with driver.session(database=database) as session:
        session.run(
            """
            MERGE (skill:Skill {name: $name})
            SET skill.description = $description, skill.purpose = $purpose
            """,
            name=name,
            description=frontmatter.get("description", ""),
            purpose=body.get("purpose", ""),
        )

        steps = body.get("steps", [])
        previous_name = None
        for step in steps:
            session.run(
                """
                MATCH (skill:Skill {name: $skill_name})
                MERGE (skill)-[:HAS_STEP]->(step:Step {skill_name: $skill_name, name: $name})
                SET step.description = $description
                """,
                skill_name=name,
                name=step["name"],
                description=step.get("description", ""),
            )
            for dependency in step.get("depends_on", []):
                session.run(
                    """
                    MATCH (step:Step {skill_name: $skill_name, name: $name})
                    MATCH (dependency:Step {skill_name: $skill_name, name: $dependency})
                    MERGE (step)-[:DEPENDS_ON]->(dependency)
                    """,
                    skill_name=name,
                    name=step["name"],
                    dependency=dependency,
                )
            if not step.get("depends_on") and previous_name is not None:
                session.run(
                    """
                    MATCH (previous:Step {skill_name: $skill_name, name: $previous_name})
                    MATCH (current:Step {skill_name: $skill_name, name: $name})
                    MERGE (previous)-[:NEXT]->(current)
                    """,
                    skill_name=name,
                    previous_name=previous_name,
                    name=step["name"],
                )
            previous_name = step["name"]

        for anti_pattern in body.get("anti_patterns", []):
            session.run(
                """
                MATCH (skill:Skill {name: $skill_name})
                MERGE (skill)-[:HAS_ANTI_PATTERN]->(:AntiPattern {text: $text})
                """,
                skill_name=name,
                text=anti_pattern,
            )

        for scenario in body.get("scenarios", []):
            session.run(
                """
                MATCH (skill:Skill {name: $skill_name})
                MERGE (skill)-[:HAS_SCENARIO]->(:Scenario {
                    need: $need, context: $context, action: $action, outcome: $outcome
                })
                """,
                skill_name=name,
                need=scenario.get("need", ""),
                context=scenario.get("context", ""),
                action=scenario.get("action", ""),
                outcome=scenario.get("outcome", ""),
            )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run scripts/push-skill-to-graph.py <skill-name>")
    name = sys.argv[1]
    skill_dir = Path(name)
    if not skill_dir.is_dir():
        raise SystemExit(f"Not a directory: {skill_dir}")

    frontmatter, body = parse_skill(skill_dir)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    try:
        push(driver, os.environ.get("NEO4J_DATABASE", "neo4j"), name, frontmatter, body)
    finally:
        driver.close()
    print(f"Pushed {name} to the graph - {len(body.get('steps', []))} steps.")


if __name__ == "__main__":
    main()
