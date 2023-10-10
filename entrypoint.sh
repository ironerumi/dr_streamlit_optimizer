#!/bin/bash
exec streamlit run app.py --server.port=$STREAMLIT_SERVER_PORT --server.address=0.0.0.0
