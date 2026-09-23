/*
 * Copyright 2026 LiveKit, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "livekit/encoded_video_frame_buffer.h"

#include <chrono>
#include <cstdlib>
#include <utility>

#include "api/video/i420_buffer.h"
#include "rtc_base/logging.h"

namespace livekit {

namespace {

int64_t NowMillis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

void EncodedRateControlState::Store(uint64_t target_bitrate_bps,
                                    double framerate_fps) {
  std::lock_guard<std::mutex> lock(mutex_);
  request_.has_request = true;
  request_.target_bitrate_bps = target_bitrate_bps;
  request_.framerate_fps = framerate_fps;
}

EncodedRateControlRequest EncodedRateControlState::Take() {
  std::lock_guard<std::mutex> lock(mutex_);
  EncodedRateControlRequest request = request_;
  request_ = EncodedRateControlRequest();
  return request;
}

void EncodedLayerSignals::RequestKeyframe(size_t layer) {
  if (layer < kMaxEncodedLayers) {
    keyframe_[layer].store(true, std::memory_order_relaxed);
  }
}

bool EncodedLayerSignals::TakeKeyframeRequest(size_t layer) {
  return layer < kMaxEncodedLayers &&
         keyframe_[layer].exchange(false, std::memory_order_relaxed);
}

void EncodedLayerSignals::StoreRate(size_t layer,
                                    uint64_t target_bitrate_bps,
                                    double framerate_fps) {
  if (layer < kMaxEncodedLayers) {
    rates_[layer].Store(target_bitrate_bps, framerate_fps);
  }
}

EncodedRateControlRequest EncodedLayerSignals::TakeRate(size_t layer) {
  if (layer >= kMaxEncodedLayers) {
    return EncodedRateControlRequest();
  }
  return rates_[layer].Take();
}

void EncodedLayerSignals::MarkWanted(size_t layer) {
  if (layer < kMaxEncodedLayers) {
    wanted_ms_[layer].store(NowMillis(), std::memory_order_relaxed);
  }
}

int64_t EncodedLayerSignals::MillisSinceWanted(size_t layer) const {
  if (layer >= kMaxEncodedLayers) {
    return -1;
  }
  const int64_t wanted = wanted_ms_[layer].load(std::memory_order_relaxed);
  return wanted == 0 ? -1 : NowMillis() - wanted;
}

EncodedVideoFrameBuffer::EncodedVideoFrameBuffer(
    int width,
    int height,
    EncodedVideoCodec codec,
    EncodedFrameType frame_type,
    webrtc::scoped_refptr<webrtc::EncodedImageBuffer> payload,
    std::shared_ptr<EncodedLayerSignals> signals,
    uint64_t sequence)
    : codec_(codec), signals_(std::move(signals)) {
  layers_.push_back(EncodedLayer{width, height, frame_type, std::move(payload),
                                 sequence});
}

EncodedVideoFrameBuffer::EncodedVideoFrameBuffer(
    EncodedVideoCodec codec,
    std::vector<EncodedLayer> layers,
    std::shared_ptr<EncodedLayerSignals> signals)
    : codec_(codec), layers_(std::move(layers)), signals_(std::move(signals)) {}

webrtc::VideoFrameBuffer::Type EncodedVideoFrameBuffer::type() const {
  return Type::kNative;
}

int EncodedVideoFrameBuffer::width() const {
  return layers_.empty() ? 0 : layers_.back().width;
}

int EncodedVideoFrameBuffer::height() const {
  return layers_.empty() ? 0 : layers_.back().height;
}

webrtc::scoped_refptr<webrtc::I420BufferInterface>
EncodedVideoFrameBuffer::ToI420() {
  // Sinks attached to a pre-encoded track (local preview, FFI color
  // conversion) convert whatever buffer they receive; the encoded payload
  // cannot be decoded here, so hand back a black frame instead of a null
  // buffer that would crash the caller.
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    RTC_LOG(LS_WARNING) << "EncodedVideoFrameBuffer::ToI420 cannot decode an "
                           "encoded access unit; returning black frames";
  }
  webrtc::scoped_refptr<webrtc::I420Buffer> buffer =
      webrtc::I420Buffer::Create(width(), height());
  webrtc::I420Buffer::SetBlack(buffer.get());
  return buffer;
}

webrtc::scoped_refptr<webrtc::VideoFrameBuffer>
EncodedVideoFrameBuffer::CropAndScale(int /* offset_x */,
                                      int /* offset_y */,
                                      int /* crop_width */,
                                      int /* crop_height */,
                                      int /* scaled_width */,
                                      int /* scaled_height */) {
  // Encoded payloads cannot be rescaled; returning the buffer unchanged
  // keeps misbehaving callers alive (the capture path never scales encoded
  // frames).
  RTC_LOG(LS_WARNING) << "EncodedVideoFrameBuffer::CropAndScale is "
                         "unsupported; returning the frame unscaled";
  return webrtc::scoped_refptr<webrtc::VideoFrameBuffer>(this);
}

size_t EncodedVideoFrameBuffer::LayerIndexFor(int width, int height) const {
  size_t best = layers_.empty() ? 0 : layers_.size() - 1;
  int64_t best_distance = -1;
  const int64_t area = static_cast<int64_t>(width) * height;
  for (size_t i = 0; i < layers_.size(); ++i) {
    const int64_t distance =
        std::llabs(static_cast<int64_t>(layers_[i].width) * layers_[i].height -
                   area);
    if (best_distance < 0 || distance < best_distance) {
      best = i;
      best_distance = distance;
    }
  }
  return best;
}

void EncodedVideoFrameBuffer::request_keyframe(size_t layer) const {
  if (signals_) {
    signals_->RequestKeyframe(layer);
  }
}

void EncodedVideoFrameBuffer::set_rate_control_request(
    size_t layer,
    uint64_t target_bitrate_bps,
    double framerate_fps) const {
  if (signals_) {
    signals_->StoreRate(layer, target_bitrate_bps, framerate_fps);
  }
}

void EncodedVideoFrameBuffer::mark_wanted(size_t layer) const {
  if (signals_) {
    signals_->MarkWanted(layer);
  }
}

EncodedVideoFrameBuffer* EncodedVideoFrameBuffer::FromNative(
    webrtc::VideoFrameBuffer* buffer) {
  if (!buffer || buffer->type() != webrtc::VideoFrameBuffer::Type::kNative) {
    return nullptr;
  }
  return dynamic_cast<EncodedVideoFrameBuffer*>(buffer);
}

}  // namespace livekit
