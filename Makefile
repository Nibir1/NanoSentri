.PHONY: help build up down logs clean shell-backend test

help:
	@echo "🚀 CertifyFlow Makefile Commands"
	@echo "================================="
	@echo "make build   : Rebuild all containers (clean build)"
	@echo "make up      : Start the system"
	@echo "make down    : Stop the system"
	@echo "make logs    : View live logs"
	@echo "make clean   : Remove containers, networks, and volumes"

# Force rebuild to ensure dependencies are fresh
build:
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Application running at http://localhost:3000"

up:
	docker-compose up -d
	@echo "✅ Application running at http://localhost:3000"
down:
	docker-compose down

logs:
	docker-compose logs -f

# Nuclear option: wipes database data too
clean:
	@echo "🧼 Cleaning up Docker containers, networks, and volumes..."
	docker-compose down -v
	docker system prune -f
	@echo "🧹 Cleaning Python bytecode, cache, and coverage files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo "✅ Clean complete."

# Debugging helper
shell-backend:
	docker-compose exec backend /bin/bash

# Run tests inside the running docker container
test:
	@echo "🧪 Running backend tests..."
	docker-compose exec backend env PYTHONPATH=. pytest --cov=app tests/ -v
	
	@echo "🧪 Running frontend tests..."
	docker-compose exec frontend npm test -- --run
	
	@echo "✅ Tests completed."