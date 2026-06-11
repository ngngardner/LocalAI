.PHONY: heartmula
heartmula:
	bash install.sh

.PHONY: run
run: heartmula
	@echo "Running heartmula..."
	bash run.sh
	@echo "heartmula run."

.PHONY: test
test: heartmula
	@echo "Testing heartmula..."
	bash test.sh
	@echo "heartmula tested."

.PHONY: protogen-clean
protogen-clean:
	$(RM) backend_pb2_grpc.py backend_pb2.py

.PHONY: clean
clean: protogen-clean
	rm -rf venv __pycache__
