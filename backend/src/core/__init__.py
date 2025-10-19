"""
Core module for interview practice application.

This module contains the core components for the interview practice application,
including conversation management, state transitions, LLM integration, and more.
"""

from .conversation import (
    setup_conversation_engine,
    ConversationState
)

from .state_transitions import (
    implement_state_transitions
)

try:
    from .llm_client import (
        setup_openai_client,
        setup_multi_llm_clients,
        implement_provider_fallback,
        create_judge_prompts,
        implement_cross_validation,
        implement_dual_llm_consensus,
        create_cost_optimizer,
        export_multi_llm_pipeline,
        run_system_validation
    )
    LLM_CLIENTS_AVAILABLE = True
except ImportError:
    print("Warning: LLM client dependencies not available")
    LLM_CLIENTS_AVAILABLE = False
    setup_openai_client = None
    setup_multi_llm_clients = None
    implement_provider_fallback = None
    create_judge_prompts = None
    implement_cross_validation = None
    implement_dual_llm_consensus = None
    create_cost_optimizer = None
    export_multi_llm_pipeline = None
    run_system_validation = None

try:
    from .llm_integration import (
        implement_structured_generation,
        generate_structured_question,
        process_candidate_response,
        validate_json_structure
    )
    LLM_INTEGRATION_AVAILABLE = True
except ImportError:
    print("Warning: LLM integration dependencies not available")
    LLM_INTEGRATION_AVAILABLE = False
    implement_structured_generation = None
    generate_structured_question = None
    process_candidate_response = None
    validate_json_structure = None

from .prompt_templates import (
    create_system_prompt_templates,
    create_user_prompt_templates,
    create_context_prompt_templates
)

from .schemas import (
    create_question_schema,
    create_response_schema
)

from .validators import (
    create_input_sanitizer,
    detect_prompt_injection,
    implement_content_filter
)

try:
    from .rag_pipeline import (
        setup_hybrid_vector_store,
        implement_adaptive_embeddings,
        create_semantic_chunking,
        implement_coarse_retrieval,
        create_fine_grained_filtering,
        build_metadata_fusion,
        setup_reranking_model,
        implement_listwise_reranking,
        optimize_reranking_batching,
        create_context_compression,
        implement_relevance_filtering,
        build_context_fusion,
        implement_aggressive_caching,
        optimize_retrieval_latency,
        create_fallback_strategies
    )
    RAG_PIPELINE_AVAILABLE = True
except ImportError:
    print("Warning: RAG pipeline dependencies not available")
    RAG_PIPELINE_AVAILABLE = False
    setup_hybrid_vector_store = None
    implement_adaptive_embeddings = None
    create_semantic_chunking = None
    implement_coarse_retrieval = None
    create_fine_grained_filtering = None
    build_metadata_fusion = None
    setup_reranking_model = None
    implement_listwise_reranking = None
    optimize_reranking_batching = None
    create_context_compression = None
    implement_relevance_filtering = None
    build_context_fusion = None
    implement_aggressive_caching = None
    optimize_retrieval_latency = None
    create_fallback_strategies = None

try:
    from .langchain_setup import (
        setup_langchain_environment,
        create_interview_agent,
        implement_state_management,
        create_question_generation_chain,
        implement_evaluation_chain,
        create_follow_up_chain,
        implement_custom_tools,
        integrate_agent_toolkit,
        implement_multi_agent_workflow,
        create_conversation_workflow,
        implement_persistent_memory,
        optimize_agent_performance
    )
    LANGCHAIN_AVAILABLE = True
except Exception as e:
    print(f"Warning: LangChain dependencies not available: {e}")
    LANGCHAIN_AVAILABLE = False
    setup_langchain_environment = None
    create_interview_agent = None
    implement_state_management = None
    create_question_generation_chain = None
    implement_evaluation_chain = None
    create_follow_up_chain = None
    implement_custom_tools = None
    integrate_agent_toolkit = None
    implement_multi_agent_workflow = None
    create_conversation_workflow = None
    implement_persistent_memory = None
    optimize_agent_performance = None

from .conversation_flow import (
    create_improvement_engine,
    build_learning_system,
    optimize_conversation_latency,
    implement_conversation_recovery,
    create_conversation_export
)

__all__ = [
    "setup_conversation_engine",
    "ConversationState",
    "implement_state_transitions",
    # LLM Clients (conditionally imported)
    "setup_openai_client",
    "setup_multi_llm_clients",
    "implement_provider_fallback",
    "create_judge_prompts",
    "implement_cross_validation",
    "implement_dual_llm_consensus",
    "create_cost_optimizer",
    "export_multi_llm_pipeline",
    "run_system_validation",
    # LLM Integration (conditionally imported)
    "implement_structured_generation",
    "generate_structured_question",
    "process_candidate_response",
    "validate_json_structure",
    "create_system_prompt_templates",
    "create_user_prompt_templates",
    "create_context_prompt_templates",
    "create_question_schema",
    "create_response_schema",
    "create_input_sanitizer",
    "detect_prompt_injection",
    "implement_content_filter",
    # RAG Pipeline (conditionally imported)
    "setup_hybrid_vector_store",
    "implement_adaptive_embeddings",
    "create_semantic_chunking",
    "implement_coarse_retrieval",
    "create_fine_grained_filtering",
    "build_metadata_fusion",
    "setup_reranking_model",
    "implement_listwise_reranking",
    "optimize_reranking_batching",
    "create_context_compression",
    "implement_relevance_filtering",
    "build_context_fusion",
    "implement_aggressive_caching",
    "optimize_retrieval_latency",
    "create_fallback_strategies",
    # LangChain (conditionally imported)
    "setup_langchain_environment",
    "create_interview_agent",
    "implement_state_management",
    "create_question_generation_chain",
    "implement_evaluation_chain",
    "create_follow_up_chain",
    "implement_custom_tools",
    "integrate_agent_toolkit",
    "implement_multi_agent_workflow",
    "create_conversation_workflow",
    "implement_persistent_memory",
    "optimize_agent_performance",
    "create_improvement_engine",
    "build_learning_system",
    "optimize_conversation_latency",
    "implement_conversation_recovery",
    "create_conversation_export"
]