from __future__ import annotations
import logging
from milvus import default_server

logging.basicConfig(level=logging.DEBUG)

with default_server:
    from pymilvus import (
        connections,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
    )

    # Milvus Lite has already started, use default_server here.
    connections.connect(host="127.0.0.1", port=default_server.listen_port)
    logging.info("Connected to Milvus")

    from pymilvus import CollectionSchema, FieldSchema, DataType

    book_id = FieldSchema(
        name="book_id",
        dtype=DataType.INT64,
        is_primary=True,
    )
    book_name = FieldSchema(
        name="book_name",
        dtype=DataType.VARCHAR,
        max_length=200,
        # The default value will be used if this field is left empty during data inserts or upserts.
        # The data type of `default_value` must be the same as that specified in `dtype`.
        default_value="Unknown",
    )
    word_count = FieldSchema(
        name="word_count",
        dtype=DataType.INT64,
        # The default value will be used if this field is left empty during data inserts or upserts.
        # The data type of `default_value` must be the same as that specified in `dtype`.
        default_value=9999,
    )
    book_intro = FieldSchema(name="book_intro", dtype=DataType.FLOAT_VECTOR, dim=2)
    schema = CollectionSchema(
        fields=[book_id, book_name, word_count, book_intro],
        description="Test book search",
        enable_dynamic_field=True,
    )
    collection_name = "book"
    from pymilvus import Collection

    collection = Collection(
        name=collection_name, schema=schema, using="default", shards_num=2
    )
    logging.info("Created collection %s", collection_name)

    # Inserts vectors in the collection:
    import random

    data = [
        [i for i in range(2000)],
        [str(i) for i in range(2000)],
        [i for i in range(10000, 12000)],
        [[random.random() for _ in range(2)] for _ in range(2000)],
    ]
    from pymilvus import Collection

    # Get an existing collection.
    mr = collection.insert(data)
    logging.info("Inserted %s entities", mr.insert_count)

    collection.load()
    logging.info("Loaded collection %s", collection_name)

    data.append([str("dy" * i) for i in range(2000)])

    # Builds indexes on the entities:
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    from pymilvus import Collection, utility

    collection.create_index(field_name="book_intro", index_params=index_params)

    utility.index_building_progress("book")

    # Loads the collection to memory and performs a vector similarity search:
    from pymilvus import Collection

    search_params = {
        "metric_type": "L2",
        "offset": 5,
        "ignore_growing": False,
        "params": {"nprobe": 10},
    }

    # Performs a vector query
    results = collection.search(
        data=[[0.1, 0.2]],
        anns_field="book_intro",
        # the sum of `offset` in `param` and `limit`
        # should be less than 16384.
        param=search_params,
        limit=10,
        expr=None,
        # set the names of the fields you want to
        # retrieve from the search result.
        output_fields=["title"],
        consistency_level="Strong",
    )

    results[0].ids

    results[0].distances

    hit = results[0][0]
    hit.entity.get("title")

    results[0].ids
    results[0].distances

    # Perform a hybrid search
    search_param = {
        "data": [[0.1, 0.2]],
        "anns_field": "book_intro",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 0},
        "limit": 10,
        "expr": "word_count <= 11000",
    }
    res = collection.search(**search_param)
    assert len(res) == 1
    hits = res[0]
    assert len(hits) == 2
    print(f"- Total hits: {len(hits)}, hits ids: {hits.ids} ")
    print(
        f"- Top1 hit id: {hits[0].id}, distance: {hits[0].distance}, score: {hits[0].score} "
    )

    # Deletes entities from the collection:
    expr = "book_id in [0,1]"

    collection.delete(expr)

    # Drops the collection:
    from pymilvus import utility

    utility.drop_collection(collection_name)
