# Contributor Guide Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write a repository-specific `AGENTS.md` contributor guide for MI-DETR.

**Architecture:** The guide is a single Markdown file at the repository root. Content is sourced from existing docs, code layout, test files, and recent git history so contributors get accurate commands and conventions.

**Tech Stack:** Markdown, Python, git

---

### Task 1: Gather repository conventions

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `tests/test_pred.py`

**Step 1: Read the existing documentation**

Run: `sed -n '1,220p' README.md`
Expected: training, validation, prediction, and ONNX commands are documented.

**Step 2: Read dependency and test examples**

Run: `sed -n '1,120p' requirements.txt`
Expected: core Python dependencies are listed without formatter or linter config.

Run: `sed -n '1,120p' tests/test_pred.py`
Expected: `unittest`-style test naming and structure are visible.

### Task 2: Draft the guide

**Files:**
- Create: `AGENTS.md`

**Step 1: Write the contributor guide**

Include:
- repository structure
- `.venv` setup and key commands
- coding style and naming
- testing guidance
- commit and PR expectations
- dataset/checkpoint/output handling notes

**Step 2: Keep it concise**

Run: `wc -w AGENTS.md`
Expected: word count stays roughly within 200-400 words.

### Task 3: Verify the result

**Files:**
- Modify: `AGENTS.md`

**Step 1: Review the rendered source**

Run: `sed -n '1,220p' AGENTS.md`
Expected: headings are clear, actionable, and repository-specific.

**Step 2: Make final adjustments if needed**

Ensure commands and paths match current repository files.
