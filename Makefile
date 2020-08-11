IMAGE=vega_fmsnew

image_build:
	docker rmi ${IMAGE}:latest --force
	docker build -t ${IMAGE}:latest .

image_run: 
	docker run --name vega_workspace -ti ${IMAGE}:latest 

container_start:
	docker start -i vega_workspace

container_rm:
	docker rm vega_workspace