import 'dart:convert';

import 'package:hive/hive.dart';

import '../config/app_config.dart';

/// Hive-backed encrypted offline queue.
///
/// When the device is offline, screening-creation and image-upload
/// requests are serialised into this queue and replayed by
/// [SyncService] once connectivity is restored.
class OfflineQueue {
  static const String _boxName = 'offlineQueue';

  Box get _box => Hive.box(_boxName);

  /// Add an item to the tail of the queue.
  Future<void> enqueue(Map<String, dynamic> item) async {
    // Enforce max queue size.
    if (_box.length >= AppConfig.maxOfflineQueueSize) {
      // Drop the oldest entry to make room.
      await _box.deleteAt(0);
    }

    final payload = {
      'queued_at': DateTime.now().toIso8601String(),
      'retries': 0,
      ...item,
    };

    await _box.add(jsonEncode(payload));
  }

  /// Peek at the head of the queue without removing it.
  Map<String, dynamic>? peek() {
    if (_box.isEmpty) return null;
    return jsonDecode(_box.getAt(0) as String) as Map<String, dynamic>;
  }

  /// Remove and return the head of the queue.
  Future<Map<String, dynamic>?> dequeue() async {
    if (_box.isEmpty) return null;
    final raw = _box.getAt(0) as String;
    await _box.deleteAt(0);
    return jsonDecode(raw) as Map<String, dynamic>;
  }

  /// Number of pending items.
  int get length => _box.length;

  /// Whether the queue is empty.
  bool get isEmpty => _box.isEmpty;

  /// Get all items (for display / debugging).
  List<Map<String, dynamic>> getAll() {
    return _box.values
        .map((raw) => jsonDecode(raw as String) as Map<String, dynamic>)
        .toList();
  }

  /// Mark an item as having failed one more retry attempt.
  /// If retries exceed [maxRetries], the item is discarded.
  Future<void> incrementRetry(int index, {int maxRetries = 5}) async {
    if (index >= _box.length) return;

    final raw = _box.getAt(index) as String;
    final item = jsonDecode(raw) as Map<String, dynamic>;
    final retries = (item['retries'] as int? ?? 0) + 1;

    if (retries > maxRetries) {
      await _box.deleteAt(index);
      return;
    }

    item['retries'] = retries;
    await _box.putAt(index, jsonEncode(item));
  }

  /// Clear the entire queue.
  Future<void> clear() async {
    await _box.clear();
  }
}
