# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from .errors import LanceDBDocumentStoreFilterError
from .utils import get_embedding_function

logger = logging.getLogger(__name__)


class LanceDBDocumentStore:
    def __init__(
        self,
        connection: Any,
        embedding: "Embeddings",
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
    ):
        """Initialize with Lance DB connection"""
        self._connection = connection
        self._embedding = embedding
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key
