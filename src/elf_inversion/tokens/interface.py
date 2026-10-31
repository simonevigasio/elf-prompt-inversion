from typing import Protocol, Any, runtime_checkable, Literal
import jax


@runtime_checkable
class PromptLayout(Protocol):
    prompt_length: int
    input_ids: jax.Array
    editable_mask: jax.Array
    def with_prompt_ids(self, ids: jax.Array) -> "PromptLayout": ... # Replace the input_ids i the current batch with the ids


@runtime_checkable
class TokenModel(Protocol):
    prompt_length: int
    # Stage B needs the special ids and the number of tokens that are known by the model
    tokenizer: Any
    embedding_table: jax.Array
    # Return the dimention of the embedding whether it is projected or encoder
    def target_dimension(self, mode: Literal["encoder", "projected"]) -> int: ...
    # Produce the codes from the batch
    def encode_token_ids(self, batch: PromptLayout, mode: Literal["encoder", "projected"]) -> jax.Array: ...
    # set the embeddings from the EmbeddingTable
    def input_embeddings(self, batch: PromptLayout) -> jax.Array: ...
    # generate the codes fromt he embeddings
    def encode_embeddings(self, embeddings: jax.Array, batch: PromptLayout, mode: Literal["encoder", "projected"]) -> jax.Array: ...
    def prepare_token_ids(self, prompt_ids) -> PromptLayout: ...
