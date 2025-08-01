# Taxonsk

## 介绍

Taxonsk 是一个 Agent 调度编排系统.

Agent-Task 程序.

## 数据库

提交变更:

```sh
$ alembic revision --autogenerate -m "..."
```

应用变更:

```sh
$ alembic upgrade head
```

回滚版本:

```sh
# 查看所有历史版本
$ alembic history

# 回滚到上一个版本
$ alembic downgrade -1

# 回滚到某个指定的版本号
$ alembic downgrade e3441e9d0285
```
