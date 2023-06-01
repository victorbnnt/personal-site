replace_main:
	kubectl replace -f 00_main.yaml --force

replace_taxifare:
	kubectl replace -f 04_taxifare.yaml --force

replace_titanic:
	kubectl replace -f 07_titanic.yaml --force

replace_raininaustralia:
	kubectl replace -f 06_rainInAustralia.yaml --force

replace_savings:
	kubectl replace -f 05_savings.yaml --force

replace_ingresses:
	kubectl replace -f ingress-jelly.yaml --force
	kubectl replace -f ingress-rewrite.yaml --force

purge_config_deployments:
	kubectl delete -f config/nginx-ingress-controller.yaml --force || true
	kubectl delete -f config/metalLb.yaml --force || true
	kubectl delete -f config/cluster-issuer.yaml --force || true
	kubectl delete -f ingress-jelly.yaml --force || true
	kubectl delete -f ingress-rewrite.yaml --force || true

purge_applications:
	kubectl delete -f 00_main.yaml --force || true
	kubectl delete -f 04_taxifare.yaml --force || true
	kubectl delete -f 07_titanic.yaml --force || true
	kubectl delete -f 06_rainInAustralia.yaml --force || true
	kubectl delete -f 05_savings.yaml --force || true

purge_all:
	kubectl delete namespace ingress-nginx --force || true
	kubectl delete namespace metallb-system --force || true

deploy_all:
	kubectl apply -f config/required_namespaces.yaml --force
	kubectl apply -f config/nginx-ingress-controller.yaml --force
	kubectl apply -f ingress-jelly.yaml --force
	kubectl apply -f ingress-rewrite.yaml --force
	kubectl apply -f 00_main.yaml --force
	kubectl apply -f 04_taxifare.yaml --force
	kubectl apply -f 07_titanic.yaml --force
	kubectl apply -f 06_rainInAustralia.yaml --force
	kubectl apply -f 05_savings.yaml --force

failing:
	kubectl apply -f config/cluster-issuer.yaml --force
	kubectl apply -f config/metalLb.yaml --force


push_git:
	git add --all
	git commit -m 'update'
	git push origin master
