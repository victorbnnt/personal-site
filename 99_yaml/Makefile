replace_main:
	kubectl replace -f 00_main.yaml --force

replace_taxifare:
	kubectl replace -f 04_taxifare.yaml --force

replace_titanic:
	kubectl replace -f 07_titanic.yaml --force

replace_raininaustralia:
	kubectl replace -f 06_rainInAustralia.yaml --force

replace_ingresses:
	kubectl replace -f ingress-jelly.yaml --force
	kubectl replace -f ingress-rewrite.yaml --force

push_git:
	git add --all
	git commit -m 'update'
	git push origin master
