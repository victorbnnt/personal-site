cyan = /bin/echo -e "\x1b[36m\#\# $1\x1b[0m"
green = /bin/echo -e "\x1b[32m\#\# $1\x1b[0m"
run_services:
	@echo ""
	@echo "######################################################################################"
	@echo "✅ Running Taxifare on port 8900"
	@nohup uvicorn 04_newyork_taxi_fare/api-dir/api.fast:app --port 8900 >/dev/null 2>&1 &
	@echo "\n✅ To start Savings, go to folder /home/victor/workplace/savings/ and execute the following command:"
	@$(call green,"nohup streamlit run app.py --server.port 8901")
	@echo "######################################################################################"
	@echo ""
