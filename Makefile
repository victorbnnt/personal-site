cyan = /bin/echo -e "\x1b[36m\#\# $1\x1b[0m"
green = /bin/echo -e "\x1b[32m\#\# $1\x1b[0m"
run_services:
	@echo ""
	@echo "######################################################################################"
	@echo "\n✅ To start Taxifare api, go to folder /home/victorbonnet/workspace/personal-site/04_newyork_taxi_fare/api-dir/ and execute the following command:"
	@$(call green,"nohup uvicorn api.fast:app --port 8900")
	@echo "\n✅ To start Savings, go to folder /home/victor/workplace/savings/ and execute the following command:"
	@$(call green,"nohup streamlit run app.py --server.port 8901")
	@echo "\n✅ To start Bluejay, follow this tutorial:"
	@$(call green,"https://carpiero.medium.com/host-a-dashboard-using-python-dash-and-linux-in-your-own-linux-server-85d891e960bc")
	@echo "######################################################################################"
	@echo ""

check_services:
	@sh check_services.sh