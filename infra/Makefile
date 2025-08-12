.PHONY: build up down shell notebook

build:
	docker-compose build

up: build
	docker-compose up -d

down:
	docker-compose down

shell:
	docker-compose exec app bash

notebook:
	docker-compose exec app poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

deps:
	docker-compose exec app poetry install

logs:
	docker-compose logs -f --tail=200