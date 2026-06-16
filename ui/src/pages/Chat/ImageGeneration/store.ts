import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { LOGGED_TOKEN_STORAGE_KEY } from '@/common/constants';
import {
  deleteAiImageAgentConversation,
  deleteAiImageGeneration,
  generateAiImage,
  getAiImageAgentConversations,
  getAiImageGenerations,
  getAiImageModels,
  saveAiImageAgentConversation,
  saveAiImageAgentGeneration,
} from '@/services';
import type {
  AiImageAgentConversation,
  AiImageGenerateParams,
  AiImageGeneration,
  AiImageModel,
} from '@/common/interface';
import Storage from '@/utils/storage';
import type {
  AgentConversation,
  AgentMessage,
  AgentRound,
  ApiMode,
  ApiProfile,
  AppSettings,
  AppMode,
  TaskParams,
  InputImage,
  MaskDraft,
  TaskRecord,
  TaskStatus,
  FavoriteCollection,
  ResponsesApiResponse,
  ResponsesOutputItem,
} from './types';
import { DEFAULT_AGENT_MAX_TOOL_ROUNDS, DEFAULT_PARAMS } from './types';
import {
  DEFAULT_SETTINGS,
  getActiveApiProfile,
  getCustomProviderDefinition,
  mergeImportedSettings,
  normalizeSettings,
  normalizeStreamPartialImages,
  validateApiProfile,
} from './lib/apiProfiles';
import { dismissAllTooltips } from './lib/tooltipDismiss';
import {
  remapImageMentionsForOrder,
  replaceImageMentionsForApi,
} from './lib/promptImageMentions';
import {
  CURRENT_THUMBNAIL_VERSION,
  putTask as dbPutTask,
  deleteTask as dbDeleteTask,
  getAllTasks,
  getAllAgentConversations,
  clearAgentConversations as dbClearAgentConversations,
  getImage,
  getImageThumbnail,
  getStoredFreshImageThumbnail,
  getAllImageIds,
  deleteImage,
  storeImage,
} from './lib/db';
import { callImageApi } from './lib/api';
import {
  callAgentConversationTitleApi,
  callAgentResponsesApi,
  callBatchImageSingle,
  parseBatchImageCallArguments,
  parseSingleImageCallArguments,
  type AgentApiResultImage,
  type BatchImageCallResult,
} from './lib/agentApi';
import {
  collectAgentRoundOutputImageSlots,
  extractAgentReferenceIds,
  getAgentCurrentReferenceId,
  getAgentGeneratedImageReferenceId,
  replaceAgentPromptImageReferencesForApi,
} from './lib/agentImageReferences';
import {
  IMAGE_FETCH_CORS_HINT,
  normalizeBase64Image,
} from './lib/imageApiShared';
import {
  getFalErrorMessage,
  getFalQueuedImageResult,
} from './lib/falAiImageApi';
import { getCustomQueuedImageResult } from './lib/openaiCompatibleImageApi';
import {
  getImageDimensions,
  validateMaskMatchesImage,
} from './lib/canvasImage';
import { orderInputImagesForMask } from './lib/mask';
import {
  getChangedParams,
  normalizeParamsForSettings,
} from './lib/paramCompatibility';
import {
  getClosestImageSizeOption,
  normalizeImageDimensionsToSize,
  normalizeImageSize,
} from './lib/size';

export const ALL_FAVORITES_COLLECTION_ID = '__all_favorites__';
export const DEFAULT_FAVORITE_COLLECTION_ID = '__default_favorites__';
export const DEFAULT_FAVORITE_COLLECTION_NAME = '默认';

// ===== Image cache =====
// 内存缓存，id → dataUrl。只保留少量最近使用图片，避免大量 4K data URL 常驻内存。

const imageCache = new Map<string, string>();
const thumbnailCache = new Map<
  string,
  {
    dataUrl: string;
    width?: number;
    height?: number;
    thumbnailVersion?: number;
  }
>();

type LoadSystemImageGenerationsOptions = {
  silent?: boolean;
};

let systemImageGenerationsLoadPromise: Promise<void> | null = null;
let systemImageGenerationsShouldToastOnError = false;
const thumbnailBackfillIds = new Map<string, 'visible' | 'background'>();
const thumbnailBackfillRunningIds = new Set<string>();
const thumbnailSubscribers = new Map<
  string,
  Set<(thumbnail: { dataUrl: string; width?: number; height?: number }) => void>
>();
let thumbnailBackfillScheduled = false;
const MAX_IMAGE_CACHE_ENTRIES = 8;
const MAX_THUMBNAIL_CACHE_ENTRIES = 80;
const MAX_THUMBNAIL_BACKFILL_CONCURRENT = 4;
const FAL_RECOVERY_POLL_MS = 10_000;
const CUSTOM_RECOVERY_POLL_MS = 10_000;
const LOCAL_RUNNING_TASK_STALE_MS = 10 * 60 * 1000;
const LOCAL_REMOTE_RUNNING_MERGE_WINDOW_MS = 60 * 1000;
const SUPPORT_PROMPT_IMAGE_THRESHOLD = 50;
const AGENT_INPUT_DRAFT_RETENTION_MS = 3 * 24 * 60 * 60 * 1000;
const AGENT_ROUND_IMAGE_MENTION_RE = /@(?:第)?(\d+)轮图(\d+)/g;
const SYSTEM_IMAGE_ASSET_RETRY_DELAYS_MS = [300, 800, 1500, 2500];
const falRecoveryTimers = new Map<string, ReturnType<typeof setTimeout>>();
const customRecoveryTimers = new Map<string, ReturnType<typeof setTimeout>>();
const openAIWatchdogTimers = new Map<string, ReturnType<typeof setTimeout>>();
const agentRoundControllers = new Map<string, AbortController>();
const hiddenSystemGenerationIds = new Set<string>();
let systemImageModelsLoadPromise: Promise<void> | null = null;
let agentConversationPersistenceReady = false;
let agentConversationMigrationPending = false;
const OPENAI_INTERRUPTED_ERROR = '请求中断';
const AGENT_STOPPED_MESSAGE = '已停止生成。';
const AGENT_CONVERSATION_TITLE_MAX_LENGTH = 28;
const ERROR_TOAST_MAX_LENGTH = 80;
type ToastType = 'info' | 'success' | 'error';
type AgentInputDraft = {
  prompt: string;
  inputImages: InputImage[];
  maskDraft: MaskDraft | null;
  maskEditorImageId: string | null;
  updatedAt?: number;
};

function hasTextValue(value?: string | null) {
  return Boolean(String(value ?? '').trim());
}

export function hasImageAgentThinkingModel(model?: AiImageModel | null) {
  if (!model || model.enabled === false) return false;
  if (hasTextValue(model.agent_model_id)) return true;
  return Boolean(
    model.upstreams?.some(
      (upstream) =>
        upstream.enabled !== false && hasTextValue(upstream.agent_model_id),
    ),
  );
}

export function hasAnyImageAgentThinkingModel(models: AiImageModel[]) {
  return models.some(hasImageAgentThinkingModel);
}

function getPreferredImageAgentThinkingModel(
  models: AiImageModel[],
  selectedModelId?: string,
) {
  const selectedModel = models.find(
    (model) => model.site_model_id === selectedModelId,
  );
  if (hasImageAgentThinkingModel(selectedModel)) return selectedModel;
  return models.find(hasImageAgentThinkingModel) ?? null;
}

function getAgentThinkingModelUnavailableMessage(state: AppState) {
  if (state.systemImageModelsLoading) return '图片模型正在加载';
  if (state.systemImageModelsError) return state.systemImageModelsError;
  return '暂无可用 Agent 思考模型，请联系管理员配置';
}

function getStoredInputImageSize(image: InputImage) {
  if (
    Number.isFinite(image.width) &&
    Number.isFinite(image.height) &&
    (image.width ?? 0) > 0 &&
    (image.height ?? 0) > 0
  ) {
    return normalizeImageDimensionsToSize(image.width!, image.height!);
  }
  return '';
}

async function getDataUrlReferenceImageSize(dataUrl: string) {
  if (!dataUrl) return '';
  try {
    const { width, height } = await getImageDimensions(dataUrl);
    return normalizeImageDimensionsToSize(width, height);
  } catch {
    return '';
  }
}

export async function getReferenceImageSizeForInputImages(
  inputImages: InputImage[],
) {
  for (const image of inputImages) {
    const storedSize = getStoredInputImageSize(image);
    if (storedSize) return storedSize;
    const dataUrlSize = await getDataUrlReferenceImageSize(image.dataUrl);
    if (dataUrlSize) return dataUrlSize;
  }
  return '';
}

async function getReferenceImageSizeForDataUrls(dataUrls: string[]) {
  for (const dataUrl of dataUrls) {
    const size = await getDataUrlReferenceImageSize(dataUrl);
    if (size) return size;
  }
  return '';
}

async function withReferenceImageSize(
  params: TaskParams,
  inputImages: InputImage[],
) {
  const referenceSize = await getReferenceImageSizeForInputImages(inputImages);
  return referenceSize ? { ...params, size: referenceSize } : params;
}

async function withReferenceDataUrlSize(
  params: TaskParams,
  dataUrls: string[],
) {
  const referenceSize = await getReferenceImageSizeForDataUrls(dataUrls);
  return referenceSize ? { ...params, size: referenceSize } : params;
}

function normalizeParamsForSystemImageModel(
  params: TaskParams,
  model?: AiImageModel | null,
): TaskParams {
  const options = model?.size_options || [];
  const qualityOptions = model?.quality_options || [];
  if (!options.length && !qualityOptions.length) return params;
  const defaultSize = model?.default_size;
  const providerAPIFormat = model?.provider_api_format;
  const quality = qualityOptions.length
    ? qualityOptions.includes(params.quality)
      ? params.quality
      : (qualityOptions.includes(model?.default_quality || '')
          ? model?.default_quality
          : qualityOptions[0]) || DEFAULT_PARAMS.quality
    : params.quality;

  if (!options.length) {
    return {
      ...params,
      quality: quality as TaskParams['quality'],
    };
  }

  const allowedSizes = new Set(options.map((option) => option.value));
  if (!params.size || params.size === 'auto') {
    const nextParams = {
      ...params,
      size: 'auto',
      quality: quality as TaskParams['quality'],
    };
    if (providerAPIFormat === 'flow2api') {
      nextParams.quality = 'auto';
    }
    return nextParams;
  }
  const normalizedSize = normalizeImageSize(params.size);
  const closestSize = getClosestImageSizeOption(normalizedSize, options)?.value;
  const size = allowedSizes.has(normalizedSize)
    ? normalizedSize
    : closestSize ||
      (defaultSize && allowedSizes.has(defaultSize)
        ? defaultSize
        : options[0]?.value || DEFAULT_PARAMS.size);

  const nextParams = {
    ...params,
    size,
    quality: quality as TaskParams['quality'],
  };
  if (providerAPIFormat === 'flow2api') {
    nextParams.quality = 'auto';
  }
  return nextParams;
}

export function getErrorToastMessage(message: string): string {
  const text = message.trim();
  if (!text) return '操作失败';

  const firstLine = text.split(/\r?\n/)[0]?.trim() ?? '';
  const separatorIndex = firstLine.search(/[：:]/);
  if (separatorIndex > 0) {
    const title = firstLine.slice(0, separatorIndex).trim();
    if (isErrorToastTitle(title)) return title;
  }

  if (firstLine.length > ERROR_TOAST_MAX_LENGTH) return '操作失败，请查看详情';
  return firstLine || '操作失败';
}

function getToastMessage(message: string, type: ToastType): string {
  return type === 'error' ? getErrorToastMessage(message) : message;
}

function isErrorToastTitle(title: string): boolean {
  return /(?:失败|错误|异常|报错|无法|不能|超时|中断|断开|请先|请输入|已达上限|不存在|已丢失)$/.test(
    title,
  );
}

const TIMEOUT_STREAMING_HINT =
  '也可尝试打开「流式传输」，并提高「请求中间步骤图像数」来维持连接。';
const TIMEOUT_PARTIAL_IMAGES_ZERO_HINT =
  '官方流式接口不发送心跳，当前「请求中间步骤图像数」为 0，连接可能因无数据传输而断开。建议提高到 2 或 3。';
const TIMEOUT_PARTIAL_IMAGES_LOW_HINT =
  '也可尝试提高「请求中间步骤图像数」来维持连接，避免长时间无数据传输导致断开。';

type TimeoutStreamingHintProfile = Pick<
  ApiProfile,
  'provider' | 'streamImages' | 'streamPartialImages'
>;

function getTimeoutStreamingHint(profile?: TimeoutStreamingHintProfile | null) {
  if (profile?.provider !== 'openai') return '';
  const partialImages =
    profile.streamPartialImages ?? DEFAULT_SETTINGS.streamPartialImages ?? 0;
  if (profile.streamImages !== true) return TIMEOUT_STREAMING_HINT;
  if (partialImages === 0) return TIMEOUT_PARTIAL_IMAGES_ZERO_HINT;
  return partialImages < 3 ? TIMEOUT_PARTIAL_IMAGES_LOW_HINT : '';
}

function createOpenAITimeoutError(
  timeoutSeconds: number,
  profile?: TimeoutStreamingHintProfile | null,
) {
  return `请求超时：超过 ${timeoutSeconds} 秒仍未完成，请稍后重试或提高超时时间。${getTimeoutStreamingHint(profile)}`;
}

export function getCachedImage(id: string): string | undefined {
  const dataUrl = imageCache.get(id);
  if (dataUrl) {
    imageCache.delete(id);
    imageCache.set(id, dataUrl);
  }
  return dataUrl;
}

function cacheImage(id: string, dataUrl: string) {
  imageCache.delete(id);
  imageCache.set(id, dataUrl);
  while (imageCache.size > MAX_IMAGE_CACHE_ENTRIES) {
    const oldestKey = imageCache.keys().next().value;
    if (oldestKey == null) break;
    imageCache.delete(oldestKey);
  }
}

function getCachedThumbnail(id: string) {
  const thumbnail = thumbnailCache.get(id);
  if (thumbnail?.thumbnailVersion === CURRENT_THUMBNAIL_VERSION) {
    thumbnailCache.delete(id);
    thumbnailCache.set(id, thumbnail);
    return thumbnail;
  }
  if (thumbnail) {
    thumbnailCache.delete(id);
  }
  return undefined;
}

function cacheThumbnail(
  id: string,
  thumbnail: {
    dataUrl: string;
    width?: number;
    height?: number;
    thumbnailVersion?: number;
  },
) {
  if (thumbnail.thumbnailVersion !== CURRENT_THUMBNAIL_VERSION) return;
  thumbnailCache.delete(id);
  thumbnailCache.set(id, thumbnail);
  while (thumbnailCache.size > MAX_THUMBNAIL_CACHE_ENTRIES) {
    const oldestKey = thumbnailCache.keys().next().value;
    if (oldestKey == null) break;
    thumbnailCache.delete(oldestKey);
  }
}

export async function ensureImageCached(
  id: string,
): Promise<string | undefined> {
  const cached = getCachedImage(id);
  if (cached) return cached;
  const rec = await getImage(id);
  if (rec) {
    cacheImage(id, rec.dataUrl);
    return rec.dataUrl;
  }
  return undefined;
}

export async function ensureImageThumbnailCached(
  id: string,
): Promise<{ dataUrl: string; width?: number; height?: number } | undefined> {
  const cached = getCachedThumbnail(id);
  if (cached) return cached;

  const rec = await getStoredFreshImageThumbnail(id);
  if (!rec?.thumbnailDataUrl) {
    scheduleThumbnailBackfill([id], 'visible');
    return undefined;
  }

  const thumbnail = {
    dataUrl: rec.thumbnailDataUrl,
    width: rec.width,
    height: rec.height,
    thumbnailVersion: rec.thumbnailVersion,
  };
  cacheThumbnail(id, thumbnail);
  return thumbnail;
}

export function subscribeImageThumbnail(
  id: string,
  callback: (thumbnail: {
    dataUrl: string;
    width?: number;
    height?: number;
  }) => void,
) {
  let subscribers = thumbnailSubscribers.get(id);
  if (!subscribers) {
    subscribers = new Set();
    thumbnailSubscribers.set(id, subscribers);
  }
  subscribers.add(callback);
  return () => {
    subscribers?.delete(callback);
    if (subscribers?.size === 0) thumbnailSubscribers.delete(id);
  };
}

function notifyImageThumbnail(
  id: string,
  thumbnail: { dataUrl: string; width?: number; height?: number },
) {
  thumbnailSubscribers.get(id)?.forEach((callback) => callback(thumbnail));
}

function scheduleThumbnailBackfill(
  ids: Iterable<string>,
  priority: 'visible' | 'background' = 'background',
) {
  for (const id of ids) {
    if (getCachedThumbnail(id) || thumbnailBackfillRunningIds.has(id)) continue;
    const currentPriority = thumbnailBackfillIds.get(id);
    if (!currentPriority || priority === 'visible')
      thumbnailBackfillIds.set(id, priority);
  }
  scheduleThumbnailBackfillTick();
}

function scheduleThumbnailBackfillTick() {
  if (thumbnailBackfillScheduled || thumbnailBackfillIds.size === 0) return;
  thumbnailBackfillScheduled = true;

  const run = () => {
    thumbnailBackfillScheduled = false;
    void processNextThumbnailBackfill();
  };

  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(run, { timeout: 2_000 });
  } else {
    globalThis.setTimeout(run, 250);
  }
}

async function processNextThumbnailBackfill() {
  if (thumbnailBackfillRunningIds.size > 0) return;

  const ids = await getNextThumbnailBackfillBatch();
  for (const id of ids) startThumbnailBackfill(id);

  if (thumbnailBackfillIds.size > 0) scheduleThumbnailBackfillTick();
}

async function getNextThumbnailBackfillBatch() {
  const candidates = getOrderedThumbnailBackfillIds().slice(
    0,
    MAX_THUMBNAIL_BACKFILL_CONCURRENT,
  );
  if (candidates.length === 0) return [];

  const sizes = await Promise.all(
    candidates.map(async (id) => {
      const image = await getImage(id);
      return { width: image?.width, height: image?.height };
    }),
  );
  const concurrency = getThumbnailConcurrencyForBatch(sizes);
  const selected = candidates.slice(0, concurrency);
  for (const id of selected) thumbnailBackfillIds.delete(id);
  return selected;
}

function getOrderedThumbnailBackfillIds() {
  const visible: string[] = [];
  const background: string[] = [];
  for (const [id, priority] of thumbnailBackfillIds) {
    if (priority === 'visible') visible.push(id);
    else background.push(id);
  }
  return [...visible, ...background];
}

function getThumbnailConcurrencyForBatch(
  sizes: Array<{ width?: number; height?: number }>,
) {
  let maxMegapixels = 0;
  for (const { width, height } of sizes) {
    if (!width || !height) return 1;
    maxMegapixels = Math.max(maxMegapixels, (width * height) / 1_000_000);
  }
  const megapixels = maxMegapixels;
  if (megapixels >= 8) return 1;
  if (megapixels >= 4) return 2;
  if (megapixels >= 2) return 3;
  return 4;
}

function startThumbnailBackfill(id: string) {
  thumbnailBackfillRunningIds.add(id);

  void (async () => {
    if (getCachedThumbnail(id)) return;

    const thumbnail = await getImageThumbnail(id);
    if (thumbnail?.thumbnailDataUrl) {
      cacheThumbnail(id, {
        dataUrl: thumbnail.thumbnailDataUrl,
        width: thumbnail.width,
        height: thumbnail.height,
        thumbnailVersion: thumbnail.thumbnailVersion,
      });
      notifyImageThumbnail(id, {
        dataUrl: thumbnail.thumbnailDataUrl,
        width: thumbnail.width,
        height: thumbnail.height,
      });
    }
  })()
    .catch(() => {
      // Keep thumbnail generation best-effort; cards remain on placeholders if it fails.
    })
    .finally(() => {
      thumbnailBackfillRunningIds.delete(id);
      scheduleThumbnailBackfillTick();
    });
}

function orderImagesWithMaskFirst(
  images: InputImage[],
  maskTargetImageId: string | null | undefined,
) {
  if (!maskTargetImageId) return images;
  const maskIdx = images.findIndex((img) => img.id === maskTargetImageId);
  if (maskIdx <= 0) return images;
  const next = [...images];
  const [maskImage] = next.splice(maskIdx, 1);
  next.unshift(maskImage);
  return next;
}

function isAgentTask(task: TaskRecord) {
  return (
    task.sourceMode === 'agent' ||
    Boolean(task.agentConversationId || task.agentRoundId)
  );
}

function showTaskCompletionNotification(..._args: [string, string]) {
  return undefined;
}

function countSuccessfulOutputImages(tasks: TaskRecord[]) {
  return tasks.reduce(
    (count, task) =>
      count +
      (task.status === 'done' && !isAgentTask(task)
        ? task.outputImages.length
        : 0),
    0,
  );
}

function skipSupportPromptForImportedData(tasks: TaskRecord[]) {
  const count = countSuccessfulOutputImages(tasks);
  useStore.setState((state) => {
    if (state.supportPromptDismissed) return {};
    if (count <= SUPPORT_PROMPT_IMAGE_THRESHOLD) {
      return { supportPromptSkippedForImportedData: false };
    }
    if (state.supportPromptOpen) return {};
    return { supportPromptSkippedForImportedData: true };
  });
}

function showSupportPromptForExistingLocalData(tasks: TaskRecord[]) {
  const count = countSuccessfulOutputImages(tasks);
  useStore.setState((state) => {
    if (state.supportPromptDismissed || state.supportPromptOpen) return {};
    if (count <= SUPPORT_PROMPT_IMAGE_THRESHOLD) {
      return { supportPromptSkippedForImportedData: false };
    }
    if (state.supportPromptSkippedForImportedData) return {};
    return { supportPromptOpen: true };
  });
}

function maybeOpenSupportPrompt(
  previousTasks: TaskRecord[],
  nextTasks: TaskRecord[],
  taskId: string,
) {
  const state = useStore.getState();
  if (
    state.supportPromptDismissed ||
    state.supportPromptOpen ||
    state.supportPromptSkippedForImportedData
  )
    return;

  const previousTask = previousTasks.find((task) => task.id === taskId);
  const nextTask = nextTasks.find((task) => task.id === taskId);
  if (
    !nextTask ||
    previousTask?.status === 'done' ||
    nextTask.status !== 'done' ||
    nextTask.outputImages.length === 0
  )
    return;

  const previousCount = countSuccessfulOutputImages(previousTasks);
  const nextCount = countSuccessfulOutputImages(nextTasks);
  if (
    previousCount <= SUPPORT_PROMPT_IMAGE_THRESHOLD &&
    nextCount > SUPPORT_PROMPT_IMAGE_THRESHOLD
  ) {
    useStore.setState({ supportPromptOpen: true });
  }
}

function normalizeStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function normalizeAgentRound(
  value: unknown,
  fallbackIndex: number,
): AgentRound | null {
  if (!value || typeof value !== 'object') return null;
  const round = value as Partial<AgentRound>;
  if (typeof round.id !== 'string' || !round.id) return null;
  if (typeof round.userMessageId !== 'string' || !round.userMessageId)
    return null;

  const status =
    round.status === 'running'
      ? 'error'
      : round.status === 'error' || round.status === 'done'
        ? round.status
        : 'done';

  return {
    id: round.id,
    index: typeof round.index === 'number' ? round.index : fallbackIndex + 1,
    parentRoundId:
      typeof round.parentRoundId === 'string' ? round.parentRoundId : null,
    userMessageId: round.userMessageId,
    ...(typeof round.assistantMessageId === 'string'
      ? { assistantMessageId: round.assistantMessageId }
      : {}),
    prompt: typeof round.prompt === 'string' ? round.prompt : '',
    inputImageIds: normalizeStringArray(round.inputImageIds),
    maskTargetImageId:
      typeof round.maskTargetImageId === 'string'
        ? round.maskTargetImageId
        : null,
    maskImageId:
      typeof round.maskImageId === 'string' ? round.maskImageId : null,
    outputTaskIds: normalizeStringArray(round.outputTaskIds),
    ...(typeof round.responseId === 'string'
      ? { responseId: round.responseId }
      : {}),
    ...(Array.isArray(round.responseOutput)
      ? { responseOutput: round.responseOutput }
      : {}),
    status,
    error:
      status === 'error'
        ? typeof round.error === 'string'
          ? round.error
          : '上次请求已中断'
        : null,
    createdAt:
      typeof round.createdAt === 'number' ? round.createdAt : Date.now(),
    finishedAt: typeof round.finishedAt === 'number' ? round.finishedAt : null,
  };
}

function normalizeAgentMessage(value: unknown): AgentMessage | null {
  if (!value || typeof value !== 'object') return null;
  const message = value as Partial<AgentMessage>;
  if (typeof message.id !== 'string' || !message.id) return null;
  if (message.role !== 'user' && message.role !== 'assistant') return null;
  if (typeof message.roundId !== 'string' || !message.roundId) return null;

  return {
    id: message.id,
    role: message.role,
    content: typeof message.content === 'string' ? message.content : '',
    roundId: message.roundId,
    ...(Array.isArray(message.inputImageIds)
      ? { inputImageIds: normalizeStringArray(message.inputImageIds) }
      : {}),
    maskTargetImageId:
      typeof message.maskTargetImageId === 'string'
        ? message.maskTargetImageId
        : null,
    maskImageId:
      typeof message.maskImageId === 'string' ? message.maskImageId : null,
    ...(Array.isArray(message.outputTaskIds)
      ? { outputTaskIds: normalizeStringArray(message.outputTaskIds) }
      : {}),
    createdAt:
      typeof message.createdAt === 'number' ? message.createdAt : Date.now(),
  };
}

function normalizeAgentConversations(value: unknown): AgentConversation[] {
  if (!Array.isArray(value)) return [];

  return value
    .filter(
      (item): item is AgentConversation =>
        Boolean(item) &&
        typeof item === 'object' &&
        typeof (item as AgentConversation).id === 'string',
    )
    .map((conversation) => {
      const normalizedRounds = Array.isArray(conversation.rounds)
        ? conversation.rounds
            .map(normalizeAgentRound)
            .filter((round): round is AgentRound => Boolean(round))
        : [];
      const hasBranchParents = normalizedRounds.some(
        (round) => round.parentRoundId,
      );
      const hasStoredActiveRound =
        typeof conversation.activeRoundId === 'string';
      const rounds =
        hasBranchParents || hasStoredActiveRound
          ? normalizedRounds
          : normalizedRounds.map((round, index) => ({
              ...round,
              parentRoundId: index > 0 ? normalizedRounds[index - 1].id : null,
            }));
      const roundIds = new Set(rounds.map((round) => round.id));
      const messages = Array.isArray(conversation.messages)
        ? conversation.messages
            .map(normalizeAgentMessage)
            .filter(
              (message): message is AgentMessage =>
                message != null && roundIds.has(message.roundId),
            )
        : [];
      return {
        id: conversation.id,
        title:
          typeof conversation.title === 'string' && conversation.title.trim()
            ? conversation.title
            : '新对话',
        activeRoundId:
          typeof conversation.activeRoundId === 'string' &&
          roundIds.has(conversation.activeRoundId)
            ? conversation.activeRoundId
            : (rounds[rounds.length - 1]?.id ?? null),
        createdAt:
          typeof conversation.createdAt === 'number'
            ? conversation.createdAt
            : Date.now(),
        updatedAt:
          typeof conversation.updatedAt === 'number'
            ? conversation.updatedAt
            : Date.now(),
        rounds,
        messages,
      };
    });
}

function mergeImportedAgentConversations(
  current: AgentConversation[],
  imported: AgentConversation[],
) {
  const merged = [...current];
  const indexes = new Map(
    merged.map((conversation, index) => [conversation.id, index]),
  );

  for (const conversation of imported) {
    const index = indexes.get(conversation.id);
    if (index == null) {
      indexes.set(conversation.id, merged.length);
      merged.push(conversation);
    } else {
      merged[index] = conversation;
    }
  }

  return merged;
}

function mergeAgentConversationsForStorage(
  stored: AgentConversation[],
  legacy: AgentConversation[],
) {
  const merged = new Map<string, AgentConversation>();
  for (const conversation of stored) merged.set(conversation.id, conversation);
  for (const conversation of legacy) {
    const existing = merged.get(conversation.id);
    if (!existing || conversation.updatedAt >= existing.updatedAt) {
      merged.set(conversation.id, conversation);
    }
  }
  return [...merged.values()].sort((a, b) => a.createdAt - b.createdAt);
}

function getPersistableResponseOutputItem(
  item: ResponsesOutputItem,
): ResponsesOutputItem {
  if (item.type !== 'image_generation_call' || item.result == null) return item;

  if (typeof item.result === 'string') {
    const { result: _result, ...rest } = item;
    return rest;
  }

  if (!isRecord(item.result)) return item;
  const {
    b64_json: _b64Json,
    base64: _base64,
    image: _image,
    data: _data,
    ...restResult
  } = item.result;
  if (Object.keys(restResult).length === 0) {
    const { result: _result, ...rest } = item;
    return rest;
  }

  return { ...item, result: restResult };
}

function getPersistableAgentConversations(
  conversations: AgentConversation[],
): AgentConversation[] {
  return conversations.map((conversation) => ({
    ...conversation,
    rounds: conversation.rounds.map((round) =>
      round.responseOutput?.length
        ? {
            ...round,
            responseOutput: round.responseOutput.map(
              getPersistableResponseOutputItem,
            ),
          }
        : round,
    ),
  }));
}

function stripPersistedAgentConversations(value: unknown): unknown {
  if (!Array.isArray(value)) return value;
  return value.map((conversation) => {
    if (!isRecord(conversation) || !Array.isArray(conversation.rounds))
      return conversation;
    return {
      ...conversation,
      rounds: conversation.rounds.map((round) => {
        if (!isRecord(round) || !Array.isArray(round.responseOutput))
          return round;
        return {
          ...round,
          responseOutput: round.responseOutput.map((item) =>
            isRecord(item)
              ? getPersistableResponseOutputItem(item as ResponsesOutputItem)
              : item,
          ),
        };
      }),
    };
  });
}

export function migratePersistedState(persistedState: unknown): unknown {
  if (!isRecord(persistedState)) return persistedState;
  return {
    ...persistedState,
    agentConversations: stripPersistedAgentConversations(
      persistedState.agentConversations,
    ),
  };
}

function normalizeFavoriteCollectionName(value: string) {
  return value.trim().replace(/\s+/g, ' ');
}

function createDefaultFavoriteCollection(now = Date.now()): FavoriteCollection {
  return {
    id: DEFAULT_FAVORITE_COLLECTION_ID,
    name: DEFAULT_FAVORITE_COLLECTION_NAME,
    createdAt: now,
    updatedAt: now,
  };
}

function normalizeFavoriteCollections(value: unknown): FavoriteCollection[] {
  const now = Date.now();
  const collections = Array.isArray(value) ? value : [];
  const normalized: FavoriteCollection[] = [];
  const ids = new Set<string>();
  for (const item of collections) {
    if (!isRecord(item)) continue;
    if (typeof item.id !== 'string' || !item.id.trim()) continue;
    const id = item.id;
    if (id === ALL_FAVORITES_COLLECTION_ID || ids.has(id)) continue;
    const name = normalizeFavoriteCollectionName(
      typeof item.name === 'string' ? item.name : '',
    );
    if (!name) continue;
    ids.add(id);
    normalized.push({
      id,
      name: name.slice(0, 60),
      createdAt: typeof item.createdAt === 'number' ? item.createdAt : now,
      updatedAt: typeof item.updatedAt === 'number' ? item.updatedAt : now,
    });
  }
  return normalized;
}

function ensureDefaultFavoriteCollection(collections: FavoriteCollection[]) {
  if (collections.length > 0) return collections;
  return [createDefaultFavoriteCollection(), ...collections];
}

/** 确保"默认"收藏夹存在（用于兜底孤立收藏任务） */
function ensureDefaultNamedCollection(collections: FavoriteCollection[]) {
  if (getDefaultNamedFavoriteCollectionId(collections)) return collections;
  return [createDefaultFavoriteCollection(), ...collections];
}

function getDefaultNamedFavoriteCollectionId(
  collections: FavoriteCollection[],
) {
  return (
    collections.find(
      (collection) => collection.id === DEFAULT_FAVORITE_COLLECTION_ID,
    )?.id ??
    collections.find(
      (collection) => collection.name === DEFAULT_FAVORITE_COLLECTION_NAME,
    )?.id ??
    null
  );
}

function resolveDefaultFavoriteCollectionId(
  collections: FavoriteCollection[],
  preferredId: unknown,
) {
  if (preferredId === null) return null;
  if (
    typeof preferredId === 'string' &&
    collections.some((collection) => collection.id === preferredId)
  )
    return preferredId;
  if (
    collections.some(
      (collection) => collection.id === DEFAULT_FAVORITE_COLLECTION_ID,
    )
  )
    return DEFAULT_FAVORITE_COLLECTION_ID;
  return collections[0]?.id ?? null;
}

function createAgentConversation(now = Date.now()): AgentConversation {
  return {
    id: genId(),
    title: '新对话',
    activeRoundId: null,
    createdAt: now,
    updatedAt: now,
    rounds: [],
    messages: [],
  };
}

function createAgentConversationTitle(prompt: string, fallbackTitle: string) {
  const title = prompt.replace(/\s+/g, ' ').trim();
  if (!title) return fallbackTitle;
  const chars = Array.from(title);
  if (chars.length <= AGENT_CONVERSATION_TITLE_MAX_LENGTH) return title;
  return `${chars.slice(0, AGENT_CONVERSATION_TITLE_MAX_LENGTH - 3).join('')}...`;
}

function isEmptyAgentConversation(conversation: AgentConversation) {
  return (
    conversation.rounds.length === 0 &&
    conversation.messages.length === 0 &&
    !conversation.activeRoundId
  );
}

function getLatestAgentConversation(conversations: AgentConversation[]) {
  return conversations.reduce<AgentConversation | null>(
    (latest, conversation) => {
      if (!latest) return conversation;
      if (conversation.updatedAt !== latest.updatedAt)
        return conversation.updatedAt > latest.updatedAt
          ? conversation
          : latest;
      return conversation.createdAt > latest.createdAt ? conversation : latest;
    },
    null,
  );
}

export function getPersistedState(state: AppState) {
  const settings = normalizeSettings(state.settings);
  return {
    settings,
    params: state.params,
    dismissedCodexCliPrompts: state.dismissedCodexCliPrompts,
    appMode: state.appMode,
    galleryInputDraft: null,
    ...(agentConversationMigrationPending && !agentConversationPersistenceReady
      ? {
          agentConversations: getPersistableAgentConversations(
            state.agentConversations,
          ),
        }
      : {}),
    activeAgentConversationId: state.activeAgentConversationId,
    agentInputDrafts: getPersistableAgentInputDrafts(state),
    agentSidebarCollapsed: state.agentSidebarCollapsed,
    agentAssetTab: state.agentAssetTab,
    agentAssetPanelCollapsed: state.agentAssetPanelCollapsed,
    favoriteCollections: state.favoriteCollections,
    defaultFavoriteCollectionId: state.defaultFavoriteCollectionId,
    selectedSystemImageModelId: state.selectedSystemImageModelId,
    supportPromptDismissed: state.supportPromptDismissed,
    supportPromptOpen: state.supportPromptOpen,
    supportPromptSkippedForImportedData:
      state.supportPromptSkippedForImportedData,
  };
}

function mapRemoteAgentConversation(
  item: AiImageAgentConversation,
): AgentConversation | null {
  try {
    const parsed = JSON.parse(item.payload) as unknown;
    const [conversation] = normalizeAgentConversations([parsed]);
    if (!conversation) return null;
    return {
      ...conversation,
      id: item.conversation_id || conversation.id,
      title: item.title || conversation.title,
      createdAt: item.created_at
        ? item.created_at * 1000
        : conversation.createdAt,
      updatedAt: item.updated_at
        ? item.updated_at * 1000
        : conversation.updatedAt,
    };
  } catch {
    return null;
  }
}

async function loadRemoteAgentConversations(): Promise<AgentConversation[]> {
  const records = (await getAiImageAgentConversations()) || [];
  return records
    .map(mapRemoteAgentConversation)
    .filter((conversation): conversation is AgentConversation =>
      Boolean(conversation),
    )
    .sort((a, b) => a.createdAt - b.createdAt);
}

async function saveRemoteAgentConversation(conversation: AgentConversation) {
  const persistable = getPersistableAgentConversation(conversation);
  await saveAiImageAgentConversation({
    conversation_id: persistable.id,
    title: persistable.title,
    payload: JSON.stringify(persistable),
  });
}

function getPersistableAgentConversation(
  conversation: AgentConversation,
): AgentConversation {
  return getPersistableAgentConversations([conversation])[0]!;
}

function mergePersistedState(
  persistedState: unknown,
  currentState: AppState,
): AppState {
  if (!persistedState || typeof persistedState !== 'object')
    return currentState;

  const persisted = persistedState as Partial<AppState>;
  const settings = normalizeSettings(
    persisted.settings ?? currentState.settings,
  );
  const hasPersistedAgentConversations = Array.isArray(
    persisted.agentConversations,
  );
  if (
    hasPersistedAgentConversations &&
    normalizeAgentConversations(persisted.agentConversations).length > 0
  ) {
    agentConversationMigrationPending = true;
  }
  const agentConversations = hasPersistedAgentConversations
    ? normalizeAgentConversations(persisted.agentConversations)
    : currentState.agentConversations;
  const activeAgentConversationId =
    typeof persisted.activeAgentConversationId === 'string' &&
    (!hasPersistedAgentConversations ||
      agentConversations.some(
        (conversation) =>
          conversation.id === persisted.activeAgentConversationId,
      ))
      ? persisted.activeAgentConversationId
      : (agentConversations[0]?.id ?? null);
  const appMode = persisted.appMode === 'agent' ? 'agent' : 'gallery';
  const galleryInputDraft = settings.persistInputOnRestart
    ? normalizeAgentInputDraft(
        persisted.galleryInputDraft ?? {
          prompt: persisted.prompt,
          inputImages: persisted.inputImages,
          maskDraft: null,
          maskEditorImageId: null,
        },
      )
    : null;
  const normalizedAgentInputDrafts = hasPersistedAgentConversations
    ? normalizeAgentInputDrafts(persisted.agentInputDrafts, agentConversations)
    : normalizeAgentInputDraftsByKey(persisted.agentInputDrafts);
  let agentInputDrafts = cleanStaleAgentInputDrafts(
    normalizedAgentInputDrafts,
    activeAgentConversationId,
  );
  if (
    appMode === 'agent' &&
    activeAgentConversationId &&
    !agentInputDrafts[activeAgentConversationId] &&
    settings.persistInputOnRestart &&
    typeof persisted.prompt === 'string'
  ) {
    agentInputDrafts = {
      ...agentInputDrafts,
      [activeAgentConversationId]: normalizeAgentInputDraft(
        {
          prompt: persisted.prompt,
          inputImages: persisted.inputImages,
          maskDraft: null,
          maskEditorImageId: null,
        },
        Date.now(),
      ),
    };
  }
  const restoredAgentDraft =
    appMode === 'agent' && activeAgentConversationId
      ? (agentInputDrafts[activeAgentConversationId] ?? null)
      : null;
  const favoriteCollections = Array.isArray(persisted.favoriteCollections)
    ? ensureDefaultFavoriteCollection(
        normalizeFavoriteCollections(persisted.favoriteCollections),
      )
    : currentState.favoriteCollections;
  const defaultFavoriteCollectionId = resolveDefaultFavoriteCollectionId(
    favoriteCollections,
    persisted.defaultFavoriteCollectionId,
  );
  return {
    ...currentState,
    ...persisted,
    settings,
    appMode,
    galleryInputDraft:
      galleryInputDraft && !isEmptyAgentInputDraft(galleryInputDraft)
        ? galleryInputDraft
        : null,
    agentConversations,
    activeAgentConversationId,
    agentInputDrafts,
    agentSidebarCollapsed: Boolean(persisted.agentSidebarCollapsed),
    agentAssetTab:
      persisted.agentAssetTab === 'references' ? 'references' : 'outputs',
    agentAssetPanelCollapsed: Boolean(persisted.agentAssetPanelCollapsed),
    favoriteCollections,
    defaultFavoriteCollectionId,
    activeFavoriteCollectionId: null,
    favoritePickerTaskIds: null,
    systemImageModels: [],
    systemImageModelsLoading: false,
    systemImageModelsLoaded: false,
    systemImageModelsError: null,
    selectedSystemImageModelId:
      typeof persisted.selectedSystemImageModelId === 'string'
        ? persisted.selectedSystemImageModelId
        : '',
    supportPromptDismissed: Boolean(persisted.supportPromptDismissed),
    supportPromptOpen: Boolean(persisted.supportPromptOpen),
    supportPromptSkippedForImportedData: Boolean(
      persisted.supportPromptSkippedForImportedData,
    ),
    prompt: restoredAgentDraft
      ? restoredAgentDraft.prompt
      : (galleryInputDraft?.prompt ?? ''),
    inputImages: restoredAgentDraft
      ? restoredAgentDraft.inputImages
      : (galleryInputDraft?.inputImages ?? []),
    maskDraft: restoredAgentDraft
      ? restoredAgentDraft.maskDraft
      : (galleryInputDraft?.maskDraft ?? null),
    maskEditorImageId: restoredAgentDraft
      ? restoredAgentDraft.maskEditorImageId
      : (galleryInputDraft?.maskEditorImageId ?? null),
  };
}

// ===== Store 类型 =====

interface AppState {
  // 模式
  appMode: AppMode;
  setAppMode: (mode: AppMode) => void;

  // 设置
  settings: AppSettings;
  setSettings: (s: Partial<AppSettings>) => void;
  systemImageModels: AiImageModel[];
  systemImageModelsLoading: boolean;
  systemImageModelsLoaded: boolean;
  systemImageModelsError: string | null;
  selectedSystemImageModelId: string;
  setSelectedSystemImageModelId: (id: string) => void;
  loadSystemImageModels: () => Promise<void>;
  loadSystemImageGenerations: (
    options?: LoadSystemImageGenerationsOptions,
  ) => Promise<void>;
  dismissedCodexCliPrompts: string[];
  dismissCodexCliPrompt: (key: string) => void;

  // 输入
  prompt: string;
  setPrompt: (p: string) => void;
  inputImages: InputImage[];
  addInputImage: (img: InputImage) => void;
  replaceInputImage: (idx: number, img: InputImage) => void;
  removeInputImage: (idx: number) => void;
  clearInputImages: () => void;
  setInputImages: (
    imgs: InputImage[],
    options?: { equivalentImageIds?: Record<string, string> },
  ) => void;
  moveInputImage: (fromIdx: number, toIdx: number) => void;
  maskDraft: MaskDraft | null;
  setMaskDraft: (draft: MaskDraft | null) => void;
  clearMaskDraft: () => void;
  maskEditorImageId: string | null;
  setMaskEditorImageId: (id: string | null) => void;
  galleryInputDraft: AgentInputDraft | null;

  // 参数
  params: TaskParams;
  setParams: (p: Partial<TaskParams>) => void;
  reusedTaskApiProfileId: string | null;
  reusedTaskApiProfileName: string | null;
  reusedTaskApiProfileMissing: boolean;
  setReusedTaskApiProfile: (
    profileId: string | null,
    missing?: boolean,
    profileName?: string | null,
  ) => void;

  // Agent
  agentConversations: AgentConversation[];
  agentConversationsLoaded: boolean;
  activeAgentConversationId: string | null;
  agentInputDrafts: Record<string, AgentInputDraft>;
  agentSidebarCollapsed: boolean;
  agentAssetTab: 'references' | 'outputs';
  agentAssetPanelCollapsed: boolean;
  agentMobileHeaderVisible: boolean;
  agentEditingRoundId: string | null;
  agentEditingConversationId: string | null;
  agentGeneratingTitleIds: Record<string, true>;
  createAgentConversation: () => string;
  setActiveAgentConversationId: (id: string | null) => void;
  setActiveAgentRoundId: (
    conversationId: string,
    roundId: string | null,
  ) => void;
  renameAgentConversation: (id: string, title: string) => void;
  deleteAgentConversation: (id: string) => void;
  setAgentSidebarCollapsed: (collapsed: boolean) => void;
  setAgentAssetTab: (tab: 'references' | 'outputs') => void;
  setAgentAssetPanelCollapsed: (collapsed: boolean) => void;
  setAgentMobileHeaderVisible: (visible: boolean) => void;
  setAgentEditingRoundId: (id: string | null) => void;
  setAgentEditingConversationId: (id: string | null) => void;

  // 任务列表
  tasks: TaskRecord[];
  setTasks: (t: TaskRecord[]) => void;
  favoriteCollections: FavoriteCollection[];
  setFavoriteCollections: (collections: FavoriteCollection[]) => void;
  defaultFavoriteCollectionId: string | null;
  setDefaultFavoriteCollectionId: (id: string | null) => void;
  activeFavoriteCollectionId: string | null;
  isManageCollectionsModalOpen: boolean;
  setActiveFavoriteCollectionId: (id: string | null) => void;
  openManageCollectionsModal: () => void;
  closeManageCollectionsModal: () => void;
  favoritePickerTaskIds: string[] | null;
  openFavoritePicker: (taskIds: string[]) => void;
  closeFavoritePicker: () => void;
  streamPreviews: Record<string, string>;
  streamPreviewSlots: Record<string, Record<string, string>>;
  setTaskStreamPreview: (
    taskId: string,
    image?: string,
    requestIndex?: number,
  ) => void;

  // 搜索和筛选
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  filterStatus: 'all' | 'running' | 'done' | 'error';
  setFilterStatus: (status: AppState['filterStatus']) => void;
  filterFavorite: boolean;
  setFilterFavorite: (f: boolean) => void;

  // 多选
  selectedTaskIds: string[];
  setSelectedTaskIds: (ids: string[] | ((prev: string[]) => string[])) => void;
  toggleTaskSelection: (id: string, force?: boolean) => void;
  clearSelection: () => void;
  selectedFavoriteCollectionIds: string[];
  setSelectedFavoriteCollectionIds: (
    ids: string[] | ((prev: string[]) => string[]),
  ) => void;
  toggleFavoriteCollectionSelection: (id: string, force?: boolean) => void;
  clearFavoriteCollectionSelection: () => void;

  // UI
  detailTaskId: string | null;
  setDetailTaskId: (id: string | null) => void;
  lightboxImageId: string | null;
  lightboxImageList: string[];
  setLightboxImageId: (id: string | null, list?: string[]) => void;
  showSettings: boolean;
  setShowSettings: (v: boolean) => void;
  supportPromptOpen: boolean;
  supportPromptDismissed: boolean;
  supportPromptSkippedForImportedData: boolean;
  setSupportPromptOpen: (v: boolean) => void;
  dismissSupportPrompt: () => void;

  // Toast
  toast: { message: string; type: ToastType } | null;
  showToast: (message: string, type?: ToastType) => void;

  // Confirm dialog
  confirmDialog: {
    title: string;
    message: string;
    checkbox?: {
      label: string;
      defaultChecked?: boolean;
      disabled?: boolean;
      tone?: 'primary' | 'danger';
    };
    confirmText?: string;
    cancelText?: string;
    showCancel?: boolean;
    buttons?: Array<{
      label: string;
      tone?: 'primary' | 'secondary' | 'danger' | 'warning';
      action: (checkboxChecked?: boolean) => void;
    }>;
    icon?: 'info' | 'copy';
    minConfirmDelayMs?: number;
    messageAlign?: 'left' | 'center';
    tone?: 'danger' | 'warning';
    action?: (checkboxChecked?: boolean) => void;
    cancelAction?: (checkboxChecked?: boolean) => void;
  } | null;
  setConfirmDialog: (d: AppState['confirmDialog']) => void;
}

function isImageReferencedByState(state: AppState, imageId: string) {
  if (state.inputImages.some((img) => img.id === imageId)) return true;
  if (state.galleryInputDraft?.inputImages.some((img) => img.id === imageId))
    return true;
  if (
    Object.values(state.agentInputDrafts).some((draft) =>
      draft.inputImages.some((img) => img.id === imageId),
    )
  )
    return true;
  if (
    state.tasks.some(
      (task) =>
        task.inputImageIds.includes(imageId) ||
        task.outputImages.includes(imageId) ||
        task.streamPartialImageIds?.includes(imageId) ||
        task.maskTargetImageId === imageId ||
        task.maskImageId === imageId,
    )
  )
    return true;
  return state.agentConversations.some(
    (conversation) =>
      conversation.rounds.some(
        (round) =>
          round.inputImageIds.includes(imageId) ||
          round.maskTargetImageId === imageId ||
          round.maskImageId === imageId,
      ) ||
      conversation.messages.some(
        (message) =>
          message.inputImageIds?.includes(imageId) ||
          message.maskTargetImageId === imageId ||
          message.maskImageId === imageId,
      ),
  );
}

export async function deleteImageIfUnreferenced(imageId: string) {
  imageCache.delete(imageId);
  thumbnailCache.delete(imageId);
  thumbnailBackfillIds.delete(imageId);
  thumbnailBackfillRunningIds.delete(imageId);
  thumbnailSubscribers.delete(imageId);
  if (isImageReferencedByState(useStore.getState(), imageId)) return;
  try {
    await deleteImage(imageId);
  } catch {
    // 清理是内存/存储优化，失败不影响替换结果。
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function normalizeInputImages(value: unknown): InputImage[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((img): InputImage | null => {
      if (!isRecord(img) || typeof img.id !== 'string') return null;
      return {
        id: img.id,
        dataUrl: typeof img.dataUrl === 'string' ? img.dataUrl : '',
      };
    })
    .filter((img): img is InputImage => img != null);
}

function normalizeMaskDraft(value: unknown): MaskDraft | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.targetImageId !== 'string' ||
    typeof value.maskDataUrl !== 'string'
  )
    return null;
  return {
    targetImageId: value.targetImageId,
    maskDataUrl: value.maskDataUrl,
    updatedAt:
      typeof value.updatedAt === 'number' ? value.updatedAt : Date.now(),
  };
}

function normalizeAgentInputDraft(
  value: unknown,
  fallbackUpdatedAt = Date.now(),
): AgentInputDraft {
  const draft = isRecord(value) ? value : {};
  const updatedAt =
    typeof draft.updatedAt === 'number' && Number.isFinite(draft.updatedAt)
      ? draft.updatedAt
      : fallbackUpdatedAt;
  return {
    prompt: typeof draft.prompt === 'string' ? draft.prompt : '',
    inputImages: normalizeInputImages(draft.inputImages),
    maskDraft: normalizeMaskDraft(draft.maskDraft),
    maskEditorImageId:
      typeof draft.maskEditorImageId === 'string'
        ? draft.maskEditorImageId
        : null,
    updatedAt,
  };
}

function normalizeAgentInputDrafts(
  value: unknown,
  conversations: AgentConversation[],
): Record<string, AgentInputDraft> {
  if (!isRecord(value)) return {};
  const conversationIds = new Set(
    conversations.map((conversation) => conversation.id),
  );
  const drafts: Record<string, AgentInputDraft> = {};
  for (const [conversationId, draft] of Object.entries(value)) {
    if (!conversationIds.has(conversationId)) continue;
    const normalized = normalizeAgentInputDraft(draft);
    if (!isEmptyAgentInputDraft(normalized))
      drafts[conversationId] = normalized;
  }
  return drafts;
}

function normalizeAgentInputDraftsByKey(
  value: unknown,
): Record<string, AgentInputDraft> {
  if (!isRecord(value)) return {};
  const drafts: Record<string, AgentInputDraft> = {};
  for (const [conversationId, draft] of Object.entries(value)) {
    const normalized = normalizeAgentInputDraft(draft);
    if (!isEmptyAgentInputDraft(normalized))
      drafts[conversationId] = normalized;
  }
  return drafts;
}

export function cleanStaleAgentInputDrafts(
  drafts: Record<string, AgentInputDraft>,
  activeConversationId: string | null,
  now = Date.now(),
) {
  const cutoff = now - AGENT_INPUT_DRAFT_RETENTION_MS;
  const next: Record<string, AgentInputDraft> = {};
  for (const [conversationId, draft] of Object.entries(drafts)) {
    if (
      conversationId === activeConversationId ||
      (draft.updatedAt ?? now) >= cutoff
    ) {
      next[conversationId] = draft;
    }
  }
  return next;
}

function clearInputDraftState(): Pick<
  AgentInputDraft,
  'prompt' | 'inputImages' | 'maskDraft' | 'maskEditorImageId'
> {
  return {
    prompt: '',
    inputImages: [],
    maskDraft: null,
    maskEditorImageId: null,
  };
}

function copyAgentInputDraft(draft: AgentInputDraft): AgentInputDraft {
  return {
    prompt: draft.prompt,
    inputImages: draft.inputImages.map((img) => ({ ...img })),
    maskDraft: draft.maskDraft ? { ...draft.maskDraft } : null,
    maskEditorImageId: draft.maskEditorImageId,
    updatedAt: draft.updatedAt ?? Date.now(),
  };
}

function getCurrentAgentInputDraft(
  state: Pick<
    AppState,
    'prompt' | 'inputImages' | 'maskDraft' | 'maskEditorImageId'
  >,
): AgentInputDraft {
  return {
    prompt: state.prompt,
    inputImages: state.inputImages,
    maskDraft: state.maskDraft,
    maskEditorImageId: state.maskEditorImageId,
    updatedAt: Date.now(),
  };
}

function isEmptyAgentInputDraft(draft: AgentInputDraft) {
  return (
    draft.prompt.length === 0 &&
    draft.inputImages.length === 0 &&
    !draft.maskDraft &&
    !draft.maskEditorImageId
  );
}

function setAgentInputDraft(
  drafts: Record<string, AgentInputDraft>,
  conversationId: string,
  draft: AgentInputDraft,
) {
  const next = { ...drafts };
  if (isEmptyAgentInputDraft(draft)) {
    delete next[conversationId];
  } else {
    next[conversationId] = copyAgentInputDraft(draft);
  }
  return next;
}

function saveActiveAgentInputDrafts(
  state: Pick<
    AppState,
    | 'appMode'
    | 'activeAgentConversationId'
    | 'agentInputDrafts'
    | 'prompt'
    | 'inputImages'
    | 'maskDraft'
    | 'maskEditorImageId'
  >,
) {
  if (state.appMode !== 'agent' || !state.activeAgentConversationId)
    return state.agentInputDrafts;
  return setAgentInputDraft(
    state.agentInputDrafts,
    state.activeAgentConversationId,
    getCurrentAgentInputDraft(state),
  );
}

function saveGalleryInputDraft(
  state: Pick<
    AppState,
    | 'appMode'
    | 'galleryInputDraft'
    | 'prompt'
    | 'inputImages'
    | 'maskDraft'
    | 'maskEditorImageId'
  >,
) {
  if (state.appMode !== 'gallery') return state.galleryInputDraft;
  const draft = getCurrentAgentInputDraft(state);
  return isEmptyAgentInputDraft(draft) ? null : copyAgentInputDraft(draft);
}

function restoreGalleryInputDraftState(
  draft: AgentInputDraft | null,
): Pick<
  AgentInputDraft,
  'prompt' | 'inputImages' | 'maskDraft' | 'maskEditorImageId'
> {
  if (!draft) return clearInputDraftState();
  return {
    prompt: draft.prompt,
    inputImages: draft.inputImages.map((img) => ({ ...img })),
    maskDraft: draft.maskDraft ? { ...draft.maskDraft } : null,
    maskEditorImageId: draft.maskEditorImageId,
  };
}

function restoreAgentInputDraftState(
  drafts: Record<string, AgentInputDraft>,
  conversationId: string | null,
): Pick<
  AgentInputDraft,
  'prompt' | 'inputImages' | 'maskDraft' | 'maskEditorImageId'
> {
  const draft = conversationId ? drafts[conversationId] : null;
  return restoreGalleryInputDraftState(draft ?? null);
}

function syncActiveInputDraft<T extends Partial<AgentInputDraft>>(
  state: AppState,
  patch: T,
): T & {
  agentInputDrafts?: Record<string, AgentInputDraft>;
  galleryInputDraft?: AgentInputDraft | null;
} {
  const draft: AgentInputDraft = {
    prompt: patch.prompt ?? state.prompt,
    inputImages: patch.inputImages ?? state.inputImages,
    maskDraft:
      patch.maskDraft !== undefined ? patch.maskDraft : state.maskDraft,
    maskEditorImageId:
      patch.maskEditorImageId !== undefined
        ? patch.maskEditorImageId
        : state.maskEditorImageId,
  };
  if (state.appMode === 'gallery') {
    return {
      ...patch,
      galleryInputDraft: isEmptyAgentInputDraft(draft)
        ? null
        : copyAgentInputDraft(draft),
    };
  }
  if (!state.activeAgentConversationId) return patch;
  return {
    ...patch,
    agentInputDrafts: setAgentInputDraft(
      state.agentInputDrafts,
      state.activeAgentConversationId,
      draft,
    ),
  };
}

function getPersistableAgentInputDrafts(state: AppState) {
  const drafts = saveActiveAgentInputDrafts(state);
  const conversationIds = new Set(
    state.agentConversations.map((conversation) => conversation.id),
  );
  const persistable: Record<string, AgentInputDraft> = {};
  for (const [conversationId, draft] of Object.entries(drafts)) {
    if (!conversationIds.has(conversationId) || isEmptyAgentInputDraft(draft))
      continue;
    persistable[conversationId] = {
      ...copyAgentInputDraft(draft),
      inputImages: draft.inputImages.map((img) => ({
        id: img.id,
        dataUrl: '',
      })),
    };
  }
  return persistable;
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Mode
      appMode: 'gallery',
      setAppMode: (appMode) => {
        if (appMode === 'gallery') {
          const state = get();
          const agentInputDrafts = saveActiveAgentInputDrafts(state);
          const galleryInputDraft = saveGalleryInputDraft(state);
          set((state) => ({
            appMode,
            agentInputDrafts,
            galleryInputDraft,
            agentMobileHeaderVisible: true,
            selectedTaskIds: [],
            selectedFavoriteCollectionIds: [],
            agentEditingRoundId: null,
            ...(state.appMode === 'agent'
              ? restoreGalleryInputDraftState(galleryInputDraft)
              : {}),
          }));
          return;
        }

        const state = get();
        const agentModel = getPreferredImageAgentThinkingModel(
          state.systemImageModels,
          state.selectedSystemImageModelId,
        );
        if (!agentModel) {
          if (
            !state.systemImageModelsLoaded &&
            !state.systemImageModelsLoading
          ) {
            void state.loadSystemImageModels();
            state.showToast('图片模型正在加载', 'info');
            return;
          }
          state.showToast(
            getAgentThinkingModelUnavailableMessage(state),
            'error',
          );
          return;
        }

        const galleryInputDraft = saveGalleryInputDraft(state);
        set((state) => ({
          appMode: 'agent',
          galleryInputDraft,
          selectedSystemImageModelId: agentModel.site_model_id,
          params: normalizeParamsForSystemImageModel(state.params, agentModel),
          agentMobileHeaderVisible: true,
          agentSidebarCollapsed: true,
          agentAssetPanelCollapsed: true,
          selectedTaskIds: [],
          selectedFavoriteCollectionIds: [],
          ...restoreAgentInputDraftState(
            state.agentInputDrafts,
            state.activeAgentConversationId,
          ),
        }));
      },

      // Settings
      settings: { ...DEFAULT_SETTINGS },
      setSettings: (s) =>
        set((st) => {
          const previous = normalizeSettings(st.settings);
          const incoming = s as Partial<AppSettings>;
          const hasLegacyOverrides =
            incoming.baseUrl !== undefined ||
            incoming.apiKey !== undefined ||
            incoming.model !== undefined ||
            incoming.timeout !== undefined ||
            incoming.apiMode !== undefined ||
            incoming.codexCli !== undefined ||
            incoming.apiProxy !== undefined ||
            incoming.streamImages !== undefined ||
            incoming.streamPartialImages !== undefined;
          const merged = normalizeSettings({ ...previous, ...incoming });
          if (hasLegacyOverrides && incoming.profiles === undefined) {
            merged.profiles = merged.profiles.map((profile) =>
              profile.id === merged.activeProfileId
                ? {
                    ...profile,
                    baseUrl: incoming.baseUrl ?? profile.baseUrl,
                    apiKey: incoming.apiKey ?? profile.apiKey,
                    model: incoming.model ?? profile.model,
                    timeout: incoming.timeout ?? profile.timeout,
                    apiMode:
                      incoming.apiMode === 'images' ||
                      incoming.apiMode === 'responses'
                        ? incoming.apiMode
                        : profile.apiMode,
                    codexCli: incoming.codexCli ?? profile.codexCli,
                    apiProxy: incoming.apiProxy ?? profile.apiProxy,
                    streamImages: incoming.streamImages ?? profile.streamImages,
                    streamPartialImages:
                      incoming.streamPartialImages ??
                      profile.streamPartialImages,
                  }
                : profile,
            );
          }
          const settings = normalizeSettings(merged);
          const shouldClearReusedProfile =
            st.reusedTaskApiProfileId &&
            settings.activeProfileId === st.reusedTaskApiProfileId;
          return {
            settings,
            ...(shouldClearReusedProfile
              ? {
                  reusedTaskApiProfileId: null,
                  reusedTaskApiProfileName: null,
                  reusedTaskApiProfileMissing: false,
                }
              : {}),
          };
        }),
      dismissedCodexCliPrompts: [],
      dismissCodexCliPrompt: (key) =>
        set((st) => ({
          dismissedCodexCliPrompts: st.dismissedCodexCliPrompts.includes(key)
            ? st.dismissedCodexCliPrompts
            : [...st.dismissedCodexCliPrompts, key],
        })),
      systemImageModels: [],
      systemImageModelsLoading: false,
      systemImageModelsLoaded: false,
      systemImageModelsError: null,
      selectedSystemImageModelId: '',
      setSelectedSystemImageModelId: (id) =>
        set((st) => {
          const selectedModel = st.systemImageModels.find(
            (model) => model.site_model_id === id,
          );
          const nextParams = normalizeParamsForSystemImageModel(
            selectedModel?.default_size
              ? { ...st.params, size: selectedModel.default_size }
              : st.params,
            selectedModel,
          );
          return {
            selectedSystemImageModelId: id,
            params: nextParams,
          };
        }),
      loadSystemImageModels: async () => {
        const state = useStore.getState();
        if (systemImageModelsLoadPromise) return systemImageModelsLoadPromise;
        if (state.systemImageModelsLoaded && state.systemImageModels.length > 0)
          return;

        useStore.setState({
          systemImageModelsLoading: true,
          systemImageModelsError: null,
        });

        systemImageModelsLoadPromise = (async () => {
          try {
            const models = (await getAiImageModels()) || [];
            const enabledModels = models.filter((model) => model.enabled);
            useStore.setState((latestState) => {
              const currentSelected = latestState.selectedSystemImageModelId;
              const selectedModel =
                enabledModels.find(
                  (model) => model.site_model_id === currentSelected,
                ) || enabledModels[0];
              return {
                systemImageModels: enabledModels,
                systemImageModelsLoading: false,
                systemImageModelsLoaded: true,
                systemImageModelsError: null,
                selectedSystemImageModelId: selectedModel?.site_model_id || '',
                params: normalizeParamsForSystemImageModel(
                  selectedModel?.default_size &&
                    latestState.params.size === DEFAULT_PARAMS.size
                    ? {
                        ...latestState.params,
                        size: selectedModel.default_size,
                      }
                    : latestState.params,
                  selectedModel,
                ),
              };
            });
          } catch (err) {
            useStore.setState({
              systemImageModelsLoading: false,
              systemImageModelsLoaded: true,
              systemImageModelsError:
                err instanceof Error
                  ? err.message
                  : '图片模型加载失败，请联系管理员检查配置',
            });
          } finally {
            systemImageModelsLoadPromise = null;
          }
        })();
        return systemImageModelsLoadPromise;
      },
      loadSystemImageGenerations: async (options) => {
        if (!options?.silent) systemImageGenerationsShouldToastOnError = true;
        if (systemImageGenerationsLoadPromise)
          return systemImageGenerationsLoadPromise;

        systemImageGenerationsLoadPromise = (async () => {
          try {
            const generations = (await getAiImageGenerations()) || [];

            const historyTasks = await Promise.all(
              generations
                .filter(
                  (generation) =>
                    !hiddenSystemGenerationIds.has(generation.generation_id),
                )
                .map(mapSystemImageGenerationToTask),
            );

            useStore.setState((state) => {
              const tasks = mergeHistoryAndLocalTasks(
                historyTasks,
                state.tasks,
              );
              const keptTaskIds = new Set(tasks.map((task) => task.id));
              void Promise.allSettled(
                state.tasks
                  .filter(
                    (task) =>
                      (task.sourceMode ?? 'gallery') === 'gallery' &&
                      !keptTaskIds.has(task.id),
                  )
                  .map((task) => dbDeleteTask(task.id)),
              );
              return { tasks };
            });
          } catch (err) {
            if (!systemImageGenerationsShouldToastOnError) {
              console.warn('图片历史刷新失败:', err);
              return;
            }
            useStore
              .getState()
              .showToast(
                err instanceof Error ? err.message : '图片历史加载失败',
                'error',
              );
          } finally {
            systemImageGenerationsLoadPromise = null;
            systemImageGenerationsShouldToastOnError = false;
          }
        })();
        return systemImageGenerationsLoadPromise;
      },

      // Input
      prompt: '',
      setPrompt: (prompt) => set((s) => syncActiveInputDraft(s, { prompt })),
      inputImages: [],
      addInputImage: (img) =>
        set((s) => {
          if (s.inputImages.find((i) => i.id === img.id)) return s;
          return syncActiveInputDraft(s, {
            inputImages: [...s.inputImages, img],
          });
        }),
      replaceInputImage: (idx, img) => {
        let removedImageId: string | null = null;
        set((s) => {
          if (idx < 0 || idx >= s.inputImages.length) return s;
          const previous = s.inputImages[idx];
          if (!previous || previous.id === img.id) return s;
          if (
            s.inputImages.some(
              (item, itemIdx) => itemIdx !== idx && item.id === img.id,
            )
          )
            return s;
          removedImageId = previous.id;
          const inputImages = s.inputImages.map((item, itemIdx) =>
            itemIdx === idx ? img : item,
          );
          const shouldClearMask = previous.id === s.maskDraft?.targetImageId;
          return syncActiveInputDraft(s, {
            inputImages,
            prompt: remapImageMentionsForOrder(
              s.prompt,
              s.inputImages,
              inputImages,
              { [previous.id]: img.id },
            ),
            ...(shouldClearMask
              ? { maskDraft: null, maskEditorImageId: null }
              : {}),
          });
        });
        if (removedImageId) void deleteImageIfUnreferenced(removedImageId);
      },
      removeInputImage: (idx) =>
        set((s) => {
          const removed = s.inputImages[idx];
          const inputImages = s.inputImages.filter((_, i) => i !== idx);
          const shouldClearMask = removed?.id === s.maskDraft?.targetImageId;
          return syncActiveInputDraft(s, {
            inputImages,
            prompt: remapImageMentionsForOrder(
              s.prompt,
              s.inputImages,
              inputImages,
            ),
            ...(shouldClearMask
              ? { maskDraft: null, maskEditorImageId: null }
              : {}),
          });
        }),
      clearInputImages: () =>
        set((s) => {
          for (const img of s.inputImages) imageCache.delete(img.id);
          return syncActiveInputDraft(s, {
            inputImages: [],
            prompt: remapImageMentionsForOrder(s.prompt, s.inputImages, []),
            maskDraft: null,
            maskEditorImageId: null,
          });
        }),
      setInputImages: (imgs, options) =>
        set((s) => {
          const inputImages = orderImagesWithMaskFirst(
            imgs,
            s.maskDraft?.targetImageId,
          );
          const shouldClearMask =
            Boolean(s.maskDraft) &&
            !inputImages.some((img) => img.id === s.maskDraft?.targetImageId);
          return syncActiveInputDraft(s, {
            inputImages,
            prompt: remapImageMentionsForOrder(
              s.prompt,
              s.inputImages,
              inputImages,
              options?.equivalentImageIds,
            ),
            ...(shouldClearMask
              ? { maskDraft: null, maskEditorImageId: null }
              : {}),
          });
        }),
      moveInputImage: (fromIdx, toIdx) =>
        set((s) => {
          const images = [...s.inputImages];
          if (fromIdx < 0 || fromIdx >= images.length) return s;
          const maskTargetImageId = s.maskDraft?.targetImageId;
          if (maskTargetImageId && images[fromIdx]?.id === maskTargetImageId)
            return s;
          const minTargetIdx =
            maskTargetImageId &&
            images.some((img) => img.id === maskTargetImageId)
              ? 1
              : 0;
          const targetIdx = Math.max(
            minTargetIdx,
            Math.min(images.length, toIdx),
          );
          const insertIdx = fromIdx < targetIdx ? targetIdx - 1 : targetIdx;
          if (insertIdx === fromIdx) return s;
          const [moved] = images.splice(fromIdx, 1);
          images.splice(insertIdx, 0, moved);
          return syncActiveInputDraft(s, {
            inputImages: images,
            prompt: remapImageMentionsForOrder(s.prompt, s.inputImages, images),
          });
        }),
      maskDraft: null,
      setMaskDraft: (maskDraft) =>
        set((s) => {
          const inputImages = orderImagesWithMaskFirst(
            s.inputImages,
            maskDraft?.targetImageId,
          );
          return syncActiveInputDraft(s, {
            maskDraft,
            inputImages,
            prompt: remapImageMentionsForOrder(
              s.prompt,
              s.inputImages,
              inputImages,
            ),
          });
        }),
      clearMaskDraft: () =>
        set((s) => syncActiveInputDraft(s, { maskDraft: null })),
      maskEditorImageId: null,
      setMaskEditorImageId: (maskEditorImageId) => {
        if (maskEditorImageId) dismissAllTooltips();
        set((s) => syncActiveInputDraft(s, { maskEditorImageId }));
      },
      galleryInputDraft: null,

      // Params
      params: { ...DEFAULT_PARAMS },
      setParams: (p) => set((s) => ({ params: { ...s.params, ...p } })),
      reusedTaskApiProfileId: null,
      reusedTaskApiProfileName: null,
      reusedTaskApiProfileMissing: false,
      setReusedTaskApiProfile: (
        profileId,
        missing = false,
        profileName = null,
      ) =>
        set({
          reusedTaskApiProfileId: profileId,
          reusedTaskApiProfileName: profileName,
          reusedTaskApiProfileMissing: missing,
        }),

      // Agent
      agentConversations: [],
      agentConversationsLoaded: false,
      activeAgentConversationId: null,
      agentInputDrafts: {},
      agentSidebarCollapsed: true,
      agentAssetTab: 'outputs',
      agentAssetPanelCollapsed: false,
      agentMobileHeaderVisible: false,
      agentEditingRoundId: null,
      agentEditingConversationId: null,
      agentGeneratingTitleIds: {},
      createAgentConversation: () => {
        const now = Date.now();
        const latestConversation = getLatestAgentConversation(
          get().agentConversations,
        );
        if (
          latestConversation &&
          isEmptyAgentConversation(latestConversation)
        ) {
          set((state) => {
            const agentInputDrafts = saveActiveAgentInputDrafts(state);
            return {
              agentConversations: state.agentConversations.map(
                (conversation) =>
                  conversation.id === latestConversation.id
                    ? { ...conversation, createdAt: now, updatedAt: now }
                    : conversation,
              ),
              activeAgentConversationId: latestConversation.id,
              agentInputDrafts,
              agentSidebarCollapsed: true,
              agentEditingRoundId: null,
              ...restoreAgentInputDraftState(
                agentInputDrafts,
                latestConversation.id,
              ),
            };
          });
          return latestConversation.id;
        }

        const conversation = createAgentConversation(now);
        set((state) => {
          const agentInputDrafts = saveActiveAgentInputDrafts(state);
          return {
            agentConversations: [...state.agentConversations, conversation],
            activeAgentConversationId: conversation.id,
            agentInputDrafts,
            agentSidebarCollapsed: true,
            agentEditingRoundId: null,
            ...restoreAgentInputDraftState(agentInputDrafts, conversation.id),
          };
        });
        return conversation.id;
      },
      setActiveAgentConversationId: (id) =>
        set((state) => {
          if (state.activeAgentConversationId === id) {
            return {
              activeAgentConversationId: id,
              agentSidebarCollapsed: true,
              agentAssetPanelCollapsed: true,
              agentEditingRoundId: null,
            };
          }
          const agentInputDrafts = saveActiveAgentInputDrafts(state);
          return {
            activeAgentConversationId: id,
            agentInputDrafts,
            agentSidebarCollapsed: true,
            agentAssetPanelCollapsed: true,
            agentEditingRoundId: null,
            ...restoreAgentInputDraftState(agentInputDrafts, id),
          };
        }),
      setActiveAgentRoundId: (conversationId, roundId) =>
        set((state) => ({
          agentConversations: state.agentConversations.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  activeRoundId: roundId,
                  updatedAt: Date.now(),
                }
              : conversation,
          ),
        })),
      renameAgentConversation: (id, title) =>
        set((state) => ({
          agentConversations: state.agentConversations.map((c) =>
            c.id === id ? { ...c, title, updatedAt: Date.now() } : c,
          ),
        })),
      deleteAgentConversation: (id) => {
        void deleteAiImageAgentConversation({ conversation_id: id }).catch(
          () => {
            useStore
              .getState()
              .showToast(
                '后端 Agent 对话删除失败，刷新后可能重新出现',
                'error',
              );
          },
        );
        set((state) => {
          const agentInputDrafts = { ...state.agentInputDrafts };
          delete agentInputDrafts[id];
          const activeDeleted = state.activeAgentConversationId === id;
          return {
            agentConversations: state.agentConversations.filter(
              (c) => c.id !== id,
            ),
            activeAgentConversationId: activeDeleted
              ? null
              : state.activeAgentConversationId,
            agentInputDrafts,
            ...(activeDeleted ? clearInputDraftState() : {}),
          };
        });
      },
      setAgentSidebarCollapsed: (agentSidebarCollapsed) =>
        set({ agentSidebarCollapsed }),
      setAgentAssetTab: (agentAssetTab) => set({ agentAssetTab }),
      setAgentAssetPanelCollapsed: (agentAssetPanelCollapsed) =>
        set({ agentAssetPanelCollapsed }),
      setAgentMobileHeaderVisible: (agentMobileHeaderVisible) =>
        set({ agentMobileHeaderVisible }),
      setAgentEditingRoundId: (agentEditingRoundId) =>
        set({ agentEditingRoundId }),
      setAgentEditingConversationId: (agentEditingConversationId) =>
        set({ agentEditingConversationId }),

      // Tasks
      tasks: [],
      setTasks: (tasks) =>
        set(() => ({
          tasks,
          ...(countSuccessfulOutputImages(tasks) <=
          SUPPORT_PROMPT_IMAGE_THRESHOLD
            ? { supportPromptSkippedForImportedData: false }
            : {}),
        })),
      favoriteCollections: [createDefaultFavoriteCollection()],
      setFavoriteCollections: (favoriteCollections) =>
        set((state) => {
          const nextCollections = ensureDefaultFavoriteCollection(
            normalizeFavoriteCollections(favoriteCollections),
          );
          return {
            favoriteCollections: nextCollections,
            defaultFavoriteCollectionId: resolveDefaultFavoriteCollectionId(
              nextCollections,
              state.defaultFavoriteCollectionId,
            ),
          };
        }),
      defaultFavoriteCollectionId: DEFAULT_FAVORITE_COLLECTION_ID,
      setDefaultFavoriteCollectionId: (defaultFavoriteCollectionId) =>
        set((state) =>
          defaultFavoriteCollectionId === null ||
          state.favoriteCollections.some(
            (collection) => collection.id === defaultFavoriteCollectionId,
          )
            ? { defaultFavoriteCollectionId }
            : state,
        ),
      activeFavoriteCollectionId: null,
      isManageCollectionsModalOpen: false,
      setActiveFavoriteCollectionId: (activeFavoriteCollectionId) =>
        set({
          activeFavoriteCollectionId,
          selectedTaskIds: [],
          selectedFavoriteCollectionIds: [],
        }),
      openManageCollectionsModal: () =>
        set({ isManageCollectionsModalOpen: true }),
      closeManageCollectionsModal: () =>
        set({ isManageCollectionsModalOpen: false }),
      favoritePickerTaskIds: null,
      openFavoritePicker: (taskIds) => {
        if (!taskIds.length) return;
        dismissAllTooltips();
        set({
          favoritePickerTaskIds: Array.from(new Set(taskIds)).filter(Boolean),
        });
      },
      closeFavoritePicker: () => set({ favoritePickerTaskIds: null }),
      streamPreviews: {},
      streamPreviewSlots: {},
      setTaskStreamPreview: (taskId, image, requestIndex = 0) =>
        set((s) => {
          if (image) {
            const slotKey = String(requestIndex);
            const currentSlots = s.streamPreviewSlots[taskId] ?? {};
            if (
              s.streamPreviews[taskId] === image &&
              currentSlots[slotKey] === image
            )
              return s;
            return {
              streamPreviews: { ...s.streamPreviews, [taskId]: image },
              streamPreviewSlots: {
                ...s.streamPreviewSlots,
                [taskId]: { ...currentSlots, [slotKey]: image },
              },
            };
          }

          if (
            !(taskId in s.streamPreviews) &&
            !(taskId in s.streamPreviewSlots)
          )
            return s;
          const next = { ...s.streamPreviews };
          const nextSlots = { ...s.streamPreviewSlots };
          delete next[taskId];
          delete nextSlots[taskId];
          return { streamPreviews: next, streamPreviewSlots: nextSlots };
        }),

      // Search & Filter
      searchQuery: '',
      setSearchQuery: (searchQuery) => set({ searchQuery }),
      filterStatus: 'all',
      setFilterStatus: (filterStatus) => set({ filterStatus }),
      filterFavorite: false,
      setFilterFavorite: (filterFavorite) =>
        set(
          filterFavorite
            ? {
                filterFavorite,
                selectedTaskIds: [],
                selectedFavoriteCollectionIds: [],
              }
            : {
                filterFavorite,
                activeFavoriteCollectionId: null,
                selectedTaskIds: [],
                selectedFavoriteCollectionIds: [],
              },
        ),

      // Selection
      selectedTaskIds: [],
      setSelectedTaskIds: (updater) =>
        set((s) => ({
          selectedTaskIds:
            typeof updater === 'function'
              ? updater(s.selectedTaskIds)
              : updater,
        })),
      toggleTaskSelection: (id, force) =>
        set((s) => {
          const isSelected = s.selectedTaskIds.includes(id);
          const shouldSelect = force !== undefined ? force : !isSelected;
          if (shouldSelect === isSelected) return s;
          return {
            selectedTaskIds: shouldSelect
              ? [...s.selectedTaskIds, id]
              : s.selectedTaskIds.filter((x) => x !== id),
          };
        }),
      clearSelection: () => set({ selectedTaskIds: [] }),
      selectedFavoriteCollectionIds: [],
      setSelectedFavoriteCollectionIds: (updater) =>
        set((s) => ({
          selectedFavoriteCollectionIds:
            typeof updater === 'function'
              ? updater(s.selectedFavoriteCollectionIds)
              : updater,
        })),
      toggleFavoriteCollectionSelection: (id, force) =>
        set((s) => {
          const isSelected = s.selectedFavoriteCollectionIds.includes(id);
          const shouldSelect = force !== undefined ? force : !isSelected;
          if (shouldSelect === isSelected) return s;
          return {
            selectedFavoriteCollectionIds: shouldSelect
              ? [...s.selectedFavoriteCollectionIds, id]
              : s.selectedFavoriteCollectionIds.filter((x) => x !== id),
          };
        }),
      clearFavoriteCollectionSelection: () =>
        set({ selectedFavoriteCollectionIds: [] }),

      // UI
      detailTaskId: null,
      setDetailTaskId: (detailTaskId) => {
        if (detailTaskId) dismissAllTooltips();
        set({ detailTaskId });
      },
      lightboxImageId: null,
      lightboxImageList: [],
      setLightboxImageId: (lightboxImageId, list) => {
        if (lightboxImageId) dismissAllTooltips();
        set({
          lightboxImageId,
          lightboxImageList: list ?? (lightboxImageId ? [lightboxImageId] : []),
        });
      },
      showSettings: false,
      setShowSettings: (showSettings) => {
        if (showSettings) dismissAllTooltips();
        set({ showSettings });
      },
      supportPromptOpen: false,
      supportPromptDismissed: false,
      supportPromptSkippedForImportedData: false,
      setSupportPromptOpen: (supportPromptOpen) => set({ supportPromptOpen }),
      dismissSupportPrompt: () =>
        set({ supportPromptOpen: false, supportPromptDismissed: true }),

      // Toast
      toast: null,
      showToast: (message, type = 'info') => {
        const toastMessage = getToastMessage(message, type);
        const toast = { message: toastMessage, type };
        set({ toast });
        setTimeout(() => {
          set((s) => (s.toast === toast ? { toast: null } : s));
        }, 3000);
      },

      // Confirm
      confirmDialog: null,
      setConfirmDialog: (confirmDialog) => {
        if (confirmDialog) dismissAllTooltips();
        set({ confirmDialog });
      },
    }),
    {
      name: 'gpt-image-playground',
      version: 2,
      migrate: (persistedState) => migratePersistedState(persistedState),
      partialize: getPersistedState,
      merge: mergePersistedState,
    },
  ),
);

let lastStoredAgentConversations = useStore.getState().agentConversations;
let agentConversationPersistRunning = false;
let agentConversationPersistQueued = false;

async function flushAgentConversationsToBackend() {
  if (agentConversationPersistRunning) {
    agentConversationPersistQueued = true;
    return;
  }

  agentConversationPersistRunning = true;
  try {
    do {
      agentConversationPersistQueued = false;
      const conversations = useStore.getState().agentConversations;
      await Promise.all(
        conversations.map((conversation) =>
          saveRemoteAgentConversation(conversation),
        ),
      );
      lastStoredAgentConversations = conversations;
    } while (
      agentConversationPersistQueued ||
      useStore.getState().agentConversations !== lastStoredAgentConversations
    );
  } finally {
    agentConversationPersistRunning = false;
  }
}

useStore.subscribe((state) => {
  if (state.agentConversations === lastStoredAgentConversations) return;
  if (!agentConversationPersistenceReady) {
    agentConversationPersistQueued = true;
    return;
  }
  void flushAgentConversationsToBackend();
});

// ===== Actions =====

let uid = 0;
function genId(): string {
  return (
    Date.now().toString(36) +
    (++uid).toString(36) +
    Math.random().toString(36).slice(2, 6)
  );
}

function getPersistableRawResponsePayload(rawResponsePayload?: string) {
  if (!rawResponsePayload) return rawResponsePayload;
  try {
    const payload = JSON.parse(rawResponsePayload) as { output?: unknown };
    if (!Array.isArray(payload.output)) return rawResponsePayload;
    const output = payload.output.map((item) =>
      isRecord(item)
        ? getPersistableResponseOutputItem(item as ResponsesOutputItem)
        : item,
    );
    return JSON.stringify({ ...payload, output }, null, 2);
  } catch {
    return rawResponsePayload;
  }
}

function getPersistableTask(task: TaskRecord): TaskRecord {
  const rawResponsePayload = getPersistableRawResponsePayload(
    task.rawResponsePayload,
  );
  return rawResponsePayload === task.rawResponsePayload
    ? task
    : { ...task, rawResponsePayload };
}

function restoreLocalGalleryTask(task: TaskRecord): TaskRecord {
  const restored = getPersistableTask(task);
  return {
    ...restored,
    rawResponsePayload: getPersistableRawResponsePayload(
      restored.rawResponsePayload,
    ),
  };
}

function putTask(task: TaskRecord): Promise<IDBValidKey> {
  return dbPutTask(getPersistableTask(task));
}

function getSystemGenerationIdForDelete(task: TaskRecord): string | null {
  if (task.systemGenerationId) return task.systemGenerationId;
  if (task.sourceMode !== 'gallery') return null;
  if (task.id.startsWith('img_')) return task.id;
  return null;
}

async function deleteRemoteImageGenerations(tasks: TaskRecord[]) {
  const generationIds = Array.from(
    new Set(
      tasks
        .map(getSystemGenerationIdForDelete)
        .filter((id): id is string => Boolean(id)),
    ),
  );
  if (generationIds.length === 0) return;

  const results = await Promise.allSettled(
    generationIds.map((generationId) =>
      deleteAiImageGeneration({ generation_id: generationId }),
    ),
  );
  const failed = results.filter((result) => result.status === 'rejected');
  if (failed.length > 0) {
    throw new Error(`有 ${failed.length} 条后端生成记录删除失败`);
  }
}

function getTaskRawImageUrlKey(task: TaskRecord): string {
  return (task.rawImageUrls ?? []).filter(Boolean).sort().join('\n');
}

function getTaskHistoryFingerprint(task: TaskRecord): string {
  if (task.status === 'running') return '';
  return [
    task.sourceMode || 'gallery',
    task.prompt.trim(),
    task.apiModel || '',
    task.params.size || '',
    task.params.quality || '',
    task.params.output_format || '',
    String(task.params.output_compression ?? ''),
    task.params.moderation || '',
    String(task.params.n || task.outputImages.length || 1),
  ].join('\n');
}

function getRunningGalleryTaskFingerprint(task: TaskRecord): string {
  if ((task.sourceMode ?? 'gallery') !== 'gallery') return '';
  if (task.status !== 'running') return '';
  if (task.inputImageIds.length > 0 || task.maskImageId) return '';
  return [
    'gallery-running',
    task.prompt.trim(),
    task.apiModel || '',
    task.params.size || '',
    task.params.quality || '',
    task.params.output_format || '',
    String(task.params.output_compression ?? ''),
    task.params.moderation || '',
    String(task.params.n || 1),
  ].join('\n');
}

function isLikelySameRunningGalleryTask(
  localTask: TaskRecord,
  historyTask: TaskRecord,
): boolean {
  if (localTask.systemGenerationId) return false;
  if (!historyTask.systemGenerationId) return false;
  const localFingerprint = getRunningGalleryTaskFingerprint(localTask);
  if (!localFingerprint) return false;
  if (localFingerprint !== getRunningGalleryTaskFingerprint(historyTask)) {
    return false;
  }
  return (
    Math.abs(historyTask.createdAt - localTask.createdAt) <=
    LOCAL_REMOTE_RUNNING_MERGE_WINDOW_MS
  );
}

function getHistoryTaskMergeKey(task: TaskRecord): string {
  return task.systemGenerationId || task.id;
}

function shouldKeepLocalTask(task: TaskRecord): boolean {
  if (task.sourceMode !== 'gallery') return true;
  return (
    task.status === 'running' &&
    Date.now() - task.createdAt < LOCAL_RUNNING_TASK_STALE_MS
  );
}

function mergeHistoryAndLocalTasks(
  historyTasks: TaskRecord[],
  currentTasks: TaskRecord[],
): TaskRecord[] {
  const historyTaskIds = new Set(historyTasks.map(getHistoryTaskMergeKey));
  const historyTaskRawImageUrlKeys = new Set(
    historyTasks.map(getTaskRawImageUrlKey).filter(Boolean),
  );
  const previousById = new Map(currentTasks.map((task) => [task.id, task]));
  const previousByGenerationId = new Map(
    currentTasks
      .filter((task) => task.systemGenerationId)
      .map((task) => [task.systemGenerationId!, task]),
  );
  const previousByRawImageUrlKey = new Map(
    currentTasks
      .map((task) => [getTaskRawImageUrlKey(task), task] as const)
      .filter(([key]) => Boolean(key)),
  );
  const previousByFingerprint = new Map(
    currentTasks
      .map((task) => [getTaskHistoryFingerprint(task), task] as const)
      .filter(([key]) => Boolean(key)),
  );
  const mergedHistoryTasks = historyTasks.map((task) => {
    const rawImageUrlKey = getTaskRawImageUrlKey(task);
    const fingerprint = getTaskHistoryFingerprint(task);
    const matchingLocalRunningTask = currentTasks.find((localTask) =>
      isLikelySameRunningGalleryTask(localTask, task),
    );
    const previous =
      previousById.get(task.id) ||
      previousByGenerationId.get(getHistoryTaskMergeKey(task)) ||
      (rawImageUrlKey
        ? previousByRawImageUrlKey.get(rawImageUrlKey)
        : undefined) ||
      (fingerprint ? previousByFingerprint.get(fingerprint) : undefined) ||
      matchingLocalRunningTask;
    return previous
      ? {
          ...task,
          id: previous.id,
          outputImages: previous.outputImages?.length
            ? previous.outputImages
            : task.outputImages,
          streamPartialImageIds: previous.streamPartialImageIds,
          streamPartialImageUrls:
            task.streamPartialImageUrls ?? previous.streamPartialImageUrls,
          systemGenerationId:
            task.systemGenerationId ?? previous.systemGenerationId,
          isFavorite: previous.isFavorite,
          favoriteCollectionIds: previous.favoriteCollectionIds,
        }
      : task;
  });
  const shouldKeepAllLocalGalleryTasks = historyTasks.length === 0;
  const localTasks = currentTasks.filter((task) => {
    const isGalleryTask = (task.sourceMode ?? 'gallery') === 'gallery';
    return (
      (shouldKeepLocalTask(task) ||
        (shouldKeepAllLocalGalleryTasks && isGalleryTask)) &&
      !historyTaskIds.has(getHistoryTaskMergeKey(task)) &&
      !historyTaskRawImageUrlKeys.has(getTaskRawImageUrlKey(task)) &&
      !historyTasks.some((historyTask) =>
        isLikelySameRunningGalleryTask(task, historyTask),
      ) &&
      !historyTasks.some(
        (historyTask) =>
          Boolean(getTaskHistoryFingerprint(historyTask)) &&
          getTaskHistoryFingerprint(historyTask) ===
            getTaskHistoryFingerprint(task),
      )
    );
  });
  return [...mergedHistoryTasks, ...localTasks].sort(
    (a, b) => b.createdAt - a.createdAt,
  );
}

function isStaleLocalGalleryTask(task: TaskRecord): boolean {
  if ((task.sourceMode ?? 'gallery') !== 'gallery') return false;
  if (task.status === 'error') return true;
  if (task.status === 'running') {
    return Date.now() - task.createdAt >= LOCAL_RUNNING_TASK_STALE_MS;
  }
  return false;
}

export function getCodexCliPromptKey(settings: AppSettings): string {
  const profile = getActiveApiProfile(settings);
  return `${profile.baseUrl}\n${profile.apiKey}`;
}

function isOpenAITask(task: TaskRecord) {
  return (task.apiProvider ?? 'openai') !== 'fal';
}

function isRunningOpenAITask(task: TaskRecord) {
  return task.status === 'running' && isOpenAITask(task);
}

function isAsyncCustomProviderTask(
  settings: AppSettings,
  provider: string,
  hasInputImages: boolean,
) {
  const customProvider = getCustomProviderDefinition(settings, provider);
  if (!customProvider?.poll) return false;
  const submitMapping =
    hasInputImages && customProvider.editSubmit
      ? customProvider.editSubmit
      : customProvider.submit;
  return Boolean(submitMapping.taskIdPath);
}

export function markInterruptedOpenAIRunningTasks(
  tasks: TaskRecord[],
  now = Date.now(),
) {
  const interruptedTasks: TaskRecord[] = [];
  const updatedTasks = tasks.map((task) => {
    if (!isRunningOpenAITask(task) || task.customTaskId) return task;

    const updated: TaskRecord = {
      ...task,
      status: 'error',
      error: OPENAI_INTERRUPTED_ERROR,
      falRecoverable: false,
      finishedAt: now,
      elapsed: Math.max(0, now - task.createdAt),
    };
    interruptedTasks.push(updated);
    return updated;
  });

  return { tasks: updatedTasks, interruptedTasks };
}

function clearOpenAIWatchdogTimer(taskId: string) {
  const timer = openAIWatchdogTimers.get(taskId);
  if (timer) clearTimeout(timer);
  openAIWatchdogTimers.delete(taskId);
}

function failOpenAITaskIfStillRunning(
  taskId: string,
  error: string,
  now = Date.now(),
) {
  const task = useStore.getState().tasks.find((item) => item.id === taskId);
  if (!task || !isRunningOpenAITask(task)) return false;

  updateTaskInStore(taskId, {
    status: 'error',
    error,
    falRecoverable: false,
    finishedAt: now,
    elapsed: Math.max(0, now - task.createdAt),
  });
  return true;
}

function scheduleOpenAIWatchdog(
  taskId: string,
  timeoutSeconds: number,
  profile?: TimeoutStreamingHintProfile | null,
) {
  clearOpenAIWatchdogTimer(taskId);
  const task = useStore.getState().tasks.find((item) => item.id === taskId);
  if (!task || !isRunningOpenAITask(task)) return;

  const timeoutMs = Math.max(0, timeoutSeconds * 1000);
  const remainingMs = Math.max(0, timeoutMs - (Date.now() - task.createdAt));
  const timer = setTimeout(() => {
    openAIWatchdogTimers.delete(taskId);
    const failed = failOpenAITaskIfStillRunning(
      taskId,
      createOpenAITimeoutError(timeoutSeconds, profile),
    );
    if (failed) useStore.getState().showToast('OpenAI 任务请求超时', 'error');
  }, remainingMs);
  openAIWatchdogTimers.set(taskId, timer);
}

export function showCodexCliPrompt(
  force = false,
  reason = '接口返回的提示词已被改写',
) {
  const state = useStore.getState();
  const settings = state.settings;
  const promptKey = getCodexCliPromptKey(settings);
  if (
    !force &&
    (settings.codexCli || state.dismissedCodexCliPrompts.includes(promptKey))
  )
    return;

  state.setConfirmDialog({
    title: '检测到 Codex CLI API',
    message: `${reason}，当前 API 来源很可能是 Codex CLI。\n\n是否开启 Codex CLI 兼容模式？开启后会禁用在此处无效的质量参数，并在 Images API 多图生成时使用并发请求，解决该 API 数量参数无效的问题。同时，提示词文本开头会加入简短的不改写要求，避免模型重写提示词，偏离原意。`,
    confirmText: '开启',
    action: () => {
      const state = useStore.getState();
      state.dismissCodexCliPrompt(promptKey);
      state.setSettings({ codexCli: true });
    },
    cancelAction: () => useStore.getState().dismissCodexCliPrompt(promptKey),
  });
}

function getFalRecoveryProfile(settings: AppSettings, task: TaskRecord) {
  const taskProfile = getTaskApiProfile(settings, task);
  if (taskProfile?.provider === 'fal') return taskProfile;
  return null;
}

function getCustomRecoveryProfile(settings: AppSettings, task: TaskRecord) {
  const provider = task.apiProvider;
  if (!provider || provider === 'openai' || provider === 'fal') return null;
  const taskProfile = getTaskApiProfile(settings, task);
  if (taskProfile?.provider === provider) return taskProfile;
  return null;
}

export function getTaskApiProfile(
  settings: AppSettings,
  task: TaskRecord,
): ApiProfile | null {
  const normalized = normalizeSettings(settings);
  const provider = task.apiProvider;

  if (!task.apiProfileId) return null;

  const byId = normalized.profiles.find(
    (profile) => profile.id === task.apiProfileId,
  );
  if (byId && (!provider || byId.provider === provider)) return byId;
  return null;
}

function createSettingsForApiProfile(
  settings: AppSettings,
  profile: ApiProfile,
): AppSettings {
  const normalized = normalizeSettings(settings);
  return normalizeSettings({
    ...normalized,
    baseUrl: profile.baseUrl,
    apiKey: profile.apiKey,
    model: profile.model,
    timeout: profile.timeout,
    apiMode: profile.apiMode,
    codexCli: profile.codexCli,
    apiProxy: profile.apiProxy,
    streamImages: profile.streamImages,
    streamPartialImages: profile.streamPartialImages,
    profiles: normalized.profiles.map((item) =>
      item.id === profile.id ? profile : item,
    ),
    activeProfileId: profile.id,
  });
}

function getReusedTaskApiProfile(
  settings: AppSettings,
  profileId: string | null,
): ApiProfile | null {
  if (!profileId) return null;
  return (
    normalizeSettings(settings).profiles.find(
      (profile) => profile.id === profileId,
    ) ?? null
  );
}

function getTaskApiProfileName(task: TaskRecord) {
  return task.apiProfileName || task.apiModel || '未知配置';
}

function isFalConnectionRecoverableError(err: unknown) {
  if (
    typeof DOMException !== 'undefined' &&
    err instanceof DOMException &&
    err.name === 'AbortError'
  )
    return true;
  const message = err instanceof Error ? err.message : String(err);
  return /abort|network|failed to fetch|fetch failed|load failed|timeout|连接|断开|中断/i.test(
    message,
  );
}

function isApiRequestNetworkError(err: unknown): boolean {
  if (err instanceof TypeError) {
    const message = err.message.toLowerCase();
    return /failed to fetch|fetch failed|load failed|networkerror|network request failed/i.test(
      message,
    );
  }
  return false;
}

function getApiModeApiName(apiMode: ApiMode) {
  return apiMode === 'responses' ? 'Responses API' : 'Image API';
}

function getApiRequestNetworkErrorHint(
  err: unknown,
  createdAt: number,
  usesApiProxy: boolean,
  profile?: Pick<
    ApiProfile,
    'provider' | 'apiMode' | 'streamImages' | 'streamPartialImages'
  > | null,
): string | null {
  if (!isApiRequestNetworkError(err)) return null;

  const elapsedSeconds = Math.max(0, (Date.now() - createdAt) / 1000);

  if (elapsedSeconds <= 15) {
    if (usesApiProxy) {
      return '提示：请求立即失败，请检查 API 代理服务是否正常运行。';
    }
    const unsupportedApiHint =
      profile?.provider === 'openai'
        ? `\n· API 不支持 ${getApiModeApiName(profile.apiMode)}`
        : '';
    return `提示：请求立即失败，可能原因：\n· API 服务器不可达或地址有误，请检查 API URL 是否正确、服务是否正常运行${unsupportedApiHint}\n· 接口不支持浏览器跨域请求，可使用本地运行版并配置 API 代理解决`;
  }

  if (elapsedSeconds >= 55 && elapsedSeconds <= 75) {
    return `提示：请求等待约 60 秒后被断开，这通常是 Nginx 等反向代理的默认超时，而非接口本身报错。可调大代理的超时时间（如 proxy_read_timeout），或降低图片尺寸/质量后重试。${getTimeoutStreamingHint(profile)}`;
  }

  if (elapsedSeconds >= 110 && elapsedSeconds <= 140) {
    return `提示：请求等待约 120 秒后被断开，这通常是 Cloudflare 等 CDN/网关的超时限制，而非接口本身报错。如果使用 Cloudflare，可考虑升级套餐或使用不经过 CDN 的直连地址。${getTimeoutStreamingHint(profile)}`;
  }

  return `提示：请求等待较长时间后被断开，通常是反向代理或网关的超时限制，而非接口本身报错。可检查代理超时设置，或降低图片尺寸/质量后重试。${getTimeoutStreamingHint(profile)}`;
}

function getRawErrorPayload(
  err: unknown,
): Pick<Partial<TaskRecord>, 'rawImageUrls' | 'rawResponsePayload'> {
  if (!(err instanceof Error)) return {};

  const rawImageUrls =
    'rawImageUrls' in err
      ? (err as { rawImageUrls?: unknown }).rawImageUrls
      : undefined;
  const rawResponsePayload =
    'rawResponsePayload' in err
      ? (err as { rawResponsePayload?: unknown }).rawResponsePayload
      : undefined;
  return {
    rawImageUrls:
      Array.isArray(rawImageUrls) && rawImageUrls.length
        ? rawImageUrls.filter((url): url is string => typeof url === 'string')
        : undefined,
    rawResponsePayload:
      typeof rawResponsePayload === 'string' ? rawResponsePayload : undefined,
  };
}

function getTaskErrorMessage(err: unknown, fallback = '生成失败'): string {
  if (err instanceof Error) return err.message;
  if (err && typeof err === 'object' && 'msg' in err) {
    const msg = (err as { msg?: unknown }).msg;
    if (typeof msg === 'string' && msg.trim()) return msg;
  }
  if (typeof err === 'string' && err.trim()) return err;
  return fallback;
}

function clearFalRecoveryTimer(taskId: string) {
  const timer = falRecoveryTimers.get(taskId);
  if (timer) clearTimeout(timer);
  falRecoveryTimers.delete(taskId);
}

function scheduleFalRecovery(taskId: string, delayMs = FAL_RECOVERY_POLL_MS) {
  if (falRecoveryTimers.has(taskId)) return;
  const timer = setTimeout(() => {
    falRecoveryTimers.delete(taskId);
    recoverFalTask(taskId);
  }, delayMs);
  falRecoveryTimers.set(taskId, timer);
}

function clearCustomRecoveryTimer(taskId: string) {
  const timer = customRecoveryTimers.get(taskId);
  if (timer) clearTimeout(timer);
  customRecoveryTimers.delete(taskId);
}

function scheduleCustomRecovery(
  taskId: string,
  delayMs = CUSTOM_RECOVERY_POLL_MS,
) {
  if (customRecoveryTimers.has(taskId)) return;
  const timer = setTimeout(() => {
    customRecoveryTimers.delete(taskId);
    recoverCustomTask(taskId);
  }, delayMs);
  customRecoveryTimers.set(taskId, timer);
}

function hasActualParams(
  params: Partial<TaskParams> | undefined,
): params is Partial<TaskParams> {
  return Boolean(params && Object.keys(params).length > 0);
}

function firstActualParams(
  paramsList: Array<Partial<TaskParams> | undefined> | undefined,
): Partial<TaskParams> | undefined {
  return paramsList?.find(hasActualParams);
}

function mapActualParamsByImage(
  outputIds: string[],
  paramsList: Array<Partial<TaskParams> | undefined> | undefined,
) {
  const mapped = paramsList?.reduce<Record<string, Partial<TaskParams>>>(
    (acc, params, index) => {
      const imgId = outputIds[index];
      if (imgId && hasActualParams(params)) acc[imgId] = params;
      return acc;
    },
    {},
  );
  return mapped && Object.keys(mapped).length > 0 ? mapped : undefined;
}

async function readImageSizeParam(
  dataUrl: string,
): Promise<Partial<TaskParams> | undefined> {
  if (typeof Image === 'undefined') return undefined;

  return new Promise((resolve) => {
    let settled = false;
    const image = new Image();
    const finish = (params: Partial<TaskParams> | undefined) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(params);
    };
    const timer = setTimeout(() => finish(undefined), 2000);
    image.onload = () => {
      if (image.naturalWidth > 0 && image.naturalHeight > 0) {
        finish({ size: `${image.naturalWidth}x${image.naturalHeight}` });
      } else {
        finish(undefined);
      }
    };
    image.onerror = () => finish(undefined);
    image.src = dataUrl;
    if (image.complete && image.naturalWidth > 0 && image.naturalHeight > 0) {
      finish({ size: `${image.naturalWidth}x${image.naturalHeight}` });
    }
  });
}

async function readImageSizeParamsList(
  images: string[],
): Promise<Array<Partial<TaskParams> | undefined>> {
  return Promise.all(images.map((image) => readImageSizeParam(image)));
}

async function resolveImageSizeParamsList(
  images: string[],
  preferred?: Array<Partial<TaskParams> | undefined>,
): Promise<Array<Partial<TaskParams> | undefined>> {
  if (preferred?.length === images.length && preferred.every(hasActualParams))
    return preferred;
  const fallback = await readImageSizeParamsList(images);
  return images.map((_, index) =>
    hasActualParams(preferred?.[index]) ? preferred?.[index] : fallback[index],
  );
}

async function completeRecoveredFalTask(
  task: TaskRecord,
  result: Awaited<ReturnType<typeof getFalQueuedImageResult>>,
) {
  const latest = useStore.getState().tasks.find((item) => item.id === task.id);
  if (!latest || latest.status === 'done') return;

  const actualParamsList = await resolveImageSizeParamsList(
    result.images,
    result.actualParamsList,
  );
  const outputIds: string[] = [];
  for (const dataUrl of result.images) {
    const imgId = await storeImage(dataUrl, 'generated');
    cacheImage(imgId, dataUrl);
    outputIds.push(imgId);
  }

  updateTaskInStore(task.id, {
    outputImages: outputIds,
    actualParams: firstActualParams(actualParamsList),
    actualParamsByImage: mapActualParamsByImage(outputIds, actualParamsList),
    revisedPromptByImage: undefined,
    status: 'done',
    error: null,
    falRecoverable: false,
    finishedAt: Date.now(),
    elapsed: Date.now() - task.createdAt,
  });
  useStore
    .getState()
    .showToast(`fal.ai 任务已恢复，共 ${outputIds.length} 张图片`, 'success');
  if (!isAgentTask(task))
    showTaskCompletionNotification(
      '图像生成完成',
      `fal.ai 任务已恢复，共 ${outputIds.length} 张图片。`,
    );
}

async function recoverFalTask(taskId: string) {
  const { settings, tasks } = useStore.getState();
  const task = tasks.find((item) => item.id === taskId);
  if (
    !task ||
    task.apiProvider !== 'fal' ||
    !task.falRequestId ||
    !task.falEndpoint ||
    task.status === 'done'
  )
    return;

  const profile = getFalRecoveryProfile(settings, task);
  if (!profile) {
    scheduleFalRecovery(taskId);
    return;
  }

  try {
    const result = await getFalQueuedImageResult(
      profile,
      task.falEndpoint,
      task.falRequestId,
      task.params,
    );
    clearFalRecoveryTimer(taskId);
    await completeRecoveredFalTask(task, result);
    return;
  } catch (err) {
    if (isFalConnectionRecoverableError(err)) {
      scheduleFalRecovery(taskId);
      return;
    }

    clearFalRecoveryTimer(taskId);
    updateTaskInStore(taskId, {
      status: 'error',
      error:
        getFalErrorMessage(err) ??
        (err instanceof Error ? err.message : String(err)),
      ...getRawErrorPayload(err),
      falRecoverable: false,
      finishedAt: Date.now(),
      elapsed: Date.now() - task.createdAt,
    });
  }
}

async function completeRecoveredCustomTask(
  task: TaskRecord,
  result: Awaited<ReturnType<typeof getCustomQueuedImageResult>>,
) {
  const latest = useStore.getState().tasks.find((item) => item.id === task.id);
  if (!latest || latest.status === 'done') return;

  const actualParamsList = await readImageSizeParamsList(result.images);
  const outputIds: string[] = [];
  for (const dataUrl of result.images) {
    const imgId = await storeImage(dataUrl, 'generated');
    cacheImage(imgId, dataUrl);
    outputIds.push(imgId);
  }

  updateTaskInStore(task.id, {
    outputImages: outputIds,
    actualParams: firstActualParams(actualParamsList),
    actualParamsByImage: mapActualParamsByImage(outputIds, actualParamsList),
    revisedPromptByImage: undefined,
    status: 'done',
    error: null,
    customRecoverable: false,
    finishedAt: Date.now(),
    elapsed: Date.now() - task.createdAt,
  });
  useStore
    .getState()
    .showToast(
      `自定义异步任务已恢复，共 ${outputIds.length} 张图片`,
      'success',
    );
  if (!isAgentTask(task))
    showTaskCompletionNotification(
      '图像生成完成',
      `自定义异步任务已恢复，共 ${outputIds.length} 张图片。`,
    );
}

async function recoverCustomTask(taskId: string) {
  const { settings, tasks } = useStore.getState();
  const task = tasks.find((item) => item.id === taskId);
  if (!task || !task.customTaskId || task.status === 'done') return;

  const profile = getCustomRecoveryProfile(settings, task);
  const customProvider = task.apiProvider
    ? getCustomProviderDefinition(settings, task.apiProvider)
    : null;
  if (!profile || !customProvider?.poll) {
    scheduleCustomRecovery(taskId);
    return;
  }

  try {
    const result = await getCustomQueuedImageResult(
      profile,
      customProvider,
      task.customTaskId,
      task.params,
    );
    clearCustomRecoveryTimer(taskId);
    await completeRecoveredCustomTask(task, result);
  } catch (err) {
    clearCustomRecoveryTimer(taskId);
    updateTaskInStore(taskId, {
      status: 'error',
      error: err instanceof Error ? err.message : String(err),
      ...getRawErrorPayload(err),
      customRecoverable: false,
      finishedAt: Date.now(),
      elapsed: Date.now() - task.createdAt,
    });
  }
}

/** 初始化：恢复设置、输入草稿、Agent 对话和本地画廊任务。 */
export async function initStore() {
  const legacyAgentConversations = normalizeAgentConversations(
    useStore.getState().agentConversations,
  );
  const storedAgentConversations = normalizeAgentConversations(
    await getAllAgentConversations(),
  );
  let remoteAgentConversations: AgentConversation[] = [];
  try {
    remoteAgentConversations = await loadRemoteAgentConversations();
  } catch (err) {
    useStore
      .getState()
      .showToast(
        err instanceof Error ? err.message : 'Agent 对话加载失败',
        'error',
      );
  }
  let loadedAgentConversations = remoteAgentConversations;
  const currentAgentConversations = normalizeAgentConversations(
    useStore.getState().agentConversations,
  );
  if (loadedAgentConversations.length === 0) {
    loadedAgentConversations = mergeAgentConversationsForStorage(
      storedAgentConversations,
      legacyAgentConversations,
    );
    loadedAgentConversations = mergeAgentConversationsForStorage(
      loadedAgentConversations,
      currentAgentConversations,
    );
  }
  const activeAgentConversationId =
    useStore.getState().activeAgentConversationId &&
    loadedAgentConversations.some(
      (conversation) =>
        conversation.id === useStore.getState().activeAgentConversationId,
    )
      ? useStore.getState().activeAgentConversationId
      : (loadedAgentConversations[0]?.id ?? null);
  if (
    loadedAgentConversations.length > 0 ||
    legacyAgentConversations.length > 0
  ) {
    useStore.setState((state) => {
      const agentInputDrafts = cleanStaleAgentInputDrafts(
        normalizeAgentInputDrafts(
          state.agentInputDrafts,
          loadedAgentConversations,
        ),
        activeAgentConversationId,
      );
      return {
        agentConversations: loadedAgentConversations,
        agentConversationsLoaded: true,
        activeAgentConversationId,
        agentInputDrafts,
        ...(state.appMode === 'agent'
          ? restoreAgentInputDraftState(
              agentInputDrafts,
              activeAgentConversationId,
            )
          : {}),
      };
    });
    if (remoteAgentConversations.length === 0) {
      await Promise.allSettled(
        loadedAgentConversations.map((conversation) =>
          saveRemoteAgentConversation(conversation),
        ),
      );
    }
    await dbClearAgentConversations();
  } else {
    useStore.setState({ agentConversationsLoaded: true });
  }
  const shouldRewritePersistedLocalState = agentConversationMigrationPending;
  agentConversationPersistenceReady = true;
  agentConversationMigrationPending = false;
  if (
    agentConversationPersistQueued ||
    useStore.getState().agentConversations !== lastStoredAgentConversations
  ) {
    await flushAgentConversationsToBackend();
  }
  if (shouldRewritePersistedLocalState) {
    useStore.setState({});
  }

  const storedGalleryTasks = (await getAllTasks()).filter(
    (task) => (task.sourceMode ?? 'gallery') === 'gallery',
  );
  const staleLocalGalleryTasks = storedGalleryTasks.filter(
    isStaleLocalGalleryTask,
  );
  if (staleLocalGalleryTasks.length > 0) {
    void Promise.allSettled(
      staleLocalGalleryTasks.map((task) => dbDeleteTask(task.id)),
    );
  }
  const localGalleryTasks = storedGalleryTasks
    .filter((task) => !isStaleLocalGalleryTask(task))
    .map(restoreLocalGalleryTask);

  const favoriteState = useStore.getState();
  const normalizedFavorites = normalizeLoadedFavoriteState(
    localGalleryTasks,
    favoriteState.favoriteCollections,
    favoriteState.defaultFavoriteCollectionId,
  );
  const localGalleryTasksWithFavorites = normalizedFavorites.tasks;
  if (normalizedFavorites.collections !== favoriteState.favoriteCollections) {
    favoriteState.setFavoriteCollections(normalizedFavorites.collections);
  }
  if (
    normalizedFavorites.defaultFavoriteCollectionId !==
    favoriteState.defaultFavoriteCollectionId
  ) {
    useStore
      .getState()
      .setDefaultFavoriteCollectionId(
        normalizedFavorites.defaultFavoriteCollectionId,
      );
  }
  useStore.setState((state) => ({
    tasks: mergeHistoryAndLocalTasks(
      state.tasks,
      localGalleryTasksWithFavorites,
    ),
  }));

  // 收集所有任务引用的图片 id
  const referencedIds = new Set<string>();
  const state = useStore.getState();
  const persistedInputImages = state.inputImages;
  const galleryInputDraft = state.galleryInputDraft;
  const agentConversations = state.agentConversations;
  const agentInputDrafts = state.agentInputDrafts;
  for (const img of persistedInputImages) referencedIds.add(img.id);
  if (galleryInputDraft) {
    for (const img of galleryInputDraft.inputImages) referencedIds.add(img.id);
  }
  for (const draft of Object.values(agentInputDrafts)) {
    for (const img of draft.inputImages) referencedIds.add(img.id);
  }
  for (const conversation of agentConversations) {
    for (const round of conversation.rounds) {
      for (const id of round.inputImageIds) referencedIds.add(id);
    }
  }
  for (const t of state.tasks) {
    addTaskReferencedImageIds(referencedIds, t);
  }

  // 只枚举 key 清理孤立图片，避免启动时把所有 4K 原图读进内存。
  const imageIds = await getAllImageIds();
  const referencedImageIds: string[] = [];
  for (const imgId of imageIds) {
    if (referencedIds.has(imgId)) {
      referencedImageIds.push(imgId);
    } else {
      await deleteImage(imgId);
    }
  }
  scheduleThumbnailBackfill(referencedImageIds);

  const restoredInputImages: InputImage[] = [];
  for (const img of persistedInputImages) {
    if (img.dataUrl) {
      restoredInputImages.push(img);
      cacheImage(img.id, img.dataUrl);
      continue;
    }
    const storedImage = await getImage(img.id);
    if (storedImage?.dataUrl) {
      restoredInputImages.push({ ...img, dataUrl: storedImage.dataUrl });
      cacheImage(img.id, storedImage.dataUrl);
    }
  }
  if (
    restoredInputImages.length !== persistedInputImages.length ||
    restoredInputImages.some(
      (img, index) => img.dataUrl !== persistedInputImages[index]?.dataUrl,
    )
  ) {
    useStore.getState().setInputImages(restoredInputImages);
  }

  if (galleryInputDraft) {
    const restoredGalleryImages: InputImage[] = [];
    for (const img of galleryInputDraft.inputImages) {
      if (img.dataUrl) {
        restoredGalleryImages.push(img);
        cacheImage(img.id, img.dataUrl);
        continue;
      }
      const storedImage = await getImage(img.id);
      if (storedImage?.dataUrl) {
        restoredGalleryImages.push({ ...img, dataUrl: storedImage.dataUrl });
        cacheImage(img.id, storedImage.dataUrl);
      }
    }
    const shouldClearMask =
      Boolean(galleryInputDraft.maskDraft) &&
      !restoredGalleryImages.some(
        (img) => img.id === galleryInputDraft.maskDraft?.targetImageId,
      );
    const restoredGalleryDraft: AgentInputDraft = {
      ...galleryInputDraft,
      inputImages: restoredGalleryImages,
      prompt: remapImageMentionsForOrder(
        galleryInputDraft.prompt,
        galleryInputDraft.inputImages,
        restoredGalleryImages,
      ),
      ...(shouldClearMask ? { maskDraft: null, maskEditorImageId: null } : {}),
    };
    const galleryDraftsChanged =
      restoredGalleryImages.length !== galleryInputDraft.inputImages.length ||
      restoredGalleryImages.some(
        (img, index) =>
          img.dataUrl !== galleryInputDraft.inputImages[index]?.dataUrl,
      ) ||
      shouldClearMask;
    if (galleryDraftsChanged) {
      const latestState = useStore.getState();
      const nextGalleryInputDraft = isEmptyAgentInputDraft(restoredGalleryDraft)
        ? null
        : restoredGalleryDraft;
      useStore.setState({
        galleryInputDraft: nextGalleryInputDraft,
        ...(latestState.appMode === 'gallery'
          ? restoreGalleryInputDraftState(nextGalleryInputDraft)
          : {}),
      });
    }
  }

  const restoredAgentInputDrafts: Record<string, AgentInputDraft> = {};
  let agentDraftsChanged = false;
  for (const [conversationId, draft] of Object.entries(agentInputDrafts)) {
    const restoredDraftImages: InputImage[] = [];
    for (const img of draft.inputImages) {
      if (img.dataUrl) {
        restoredDraftImages.push(img);
        cacheImage(img.id, img.dataUrl);
        continue;
      }
      const storedImage = await getImage(img.id);
      if (storedImage?.dataUrl) {
        restoredDraftImages.push({ ...img, dataUrl: storedImage.dataUrl });
        cacheImage(img.id, storedImage.dataUrl);
      }
    }

    const shouldClearMask =
      Boolean(draft.maskDraft) &&
      !restoredDraftImages.some(
        (img) => img.id === draft.maskDraft?.targetImageId,
      );
    const restoredDraft: AgentInputDraft = {
      ...draft,
      inputImages: restoredDraftImages,
      prompt: remapImageMentionsForOrder(
        draft.prompt,
        draft.inputImages,
        restoredDraftImages,
      ),
      ...(shouldClearMask ? { maskDraft: null, maskEditorImageId: null } : {}),
    };
    if (!isEmptyAgentInputDraft(restoredDraft))
      restoredAgentInputDrafts[conversationId] = restoredDraft;
    if (
      restoredDraftImages.length !== draft.inputImages.length ||
      restoredDraftImages.some(
        (img, index) => img.dataUrl !== draft.inputImages[index]?.dataUrl,
      ) ||
      shouldClearMask
    ) {
      agentDraftsChanged = true;
    }
  }
  if (agentDraftsChanged) {
    const latestState = useStore.getState();
    useStore.setState({
      agentInputDrafts: restoredAgentInputDrafts,
      ...(latestState.appMode === 'agent'
        ? restoreAgentInputDraftState(
            restoredAgentInputDrafts,
            latestState.activeAgentConversationId,
          )
        : {}),
    });
  }
}

/** 提交新任务 */
export async function submitTask(
  options: {
    allowFullMask?: boolean;
    useCurrentApiProfileWhenReusedMissing?: boolean;
  } = {},
) {
  const {
    settings,
    prompt,
    inputImages,
    maskDraft,
    params,
    reusedTaskApiProfileId,
    reusedTaskApiProfileName,
    reusedTaskApiProfileMissing,
    showToast,
    setConfirmDialog,
    selectedSystemImageModelId,
    systemImageModels,
    systemImageModelsLoading,
    systemImageModelsError,
  } = useStore.getState();
  const appMode = useStore.getState().appMode;

  const normalizedSettings = normalizeSettings(settings);
  let activeProfile = getActiveApiProfile(settings);
  let requestSettings = createSettingsForApiProfile(
    normalizedSettings,
    activeProfile,
  );
  if (
    normalizedSettings.reuseTaskApiProfileTemporarily &&
    (reusedTaskApiProfileId || reusedTaskApiProfileMissing)
  ) {
    const reusedProfile = getReusedTaskApiProfile(
      normalizedSettings,
      reusedTaskApiProfileId,
    );
    if (!reusedProfile) {
      if (options.useCurrentApiProfileWhenReusedMissing) {
        useStore.getState().setReusedTaskApiProfile(null);
      } else {
        setConfirmDialog({
          title: '找不到 API 配置',
          message: `找不到复用任务所使用的 API 配置「${reusedTaskApiProfileName || '未知配置'}」，要使用当前的 API 配置「${activeProfile.name}」提交任务吗？`,
          confirmText: '使用当前配置提交',
          cancelText: '放弃提交',
          action: () => {
            void submitTask({
              ...options,
              useCurrentApiProfileWhenReusedMissing: true,
            });
          },
        });
        return;
      }
    } else {
      activeProfile = reusedProfile;
      requestSettings = createSettingsForApiProfile(
        normalizedSettings,
        reusedProfile,
      );
    }
  }

  if (appMode === 'agent' && validateApiProfile(activeProfile)) {
    showToast(
      `请先完善请求 API 配置：${validateApiProfile(activeProfile)}`,
      'error',
    );
    useStore.getState().setShowSettings(true);
    return;
  }

  const selectedSystemImageModel =
    appMode === 'gallery'
      ? systemImageModels.find(
          (model) => model.site_model_id === selectedSystemImageModelId,
        ) || systemImageModels[0]
      : null;
  if (appMode === 'gallery' && !selectedSystemImageModel) {
    showToast(
      systemImageModelsLoading
        ? '图片模型正在加载，请稍后再试'
        : systemImageModelsError || '暂无可用图片模型，请联系管理员检查配置',
      'error',
    );
    if (!systemImageModelsLoading)
      void useStore.getState().loadSystemImageModels();
    return;
  }

  if (!prompt.trim()) {
    showToast('请输入提示词', 'error');
    return;
  }

  let orderedInputImages = inputImages;
  let maskImageId: string | null = null;
  let maskTargetImageId: string | null = null;

  if (maskDraft) {
    try {
      orderedInputImages = orderInputImagesForMask(
        inputImages,
        maskDraft.targetImageId,
      );
      const coverage = await validateMaskMatchesImage(
        maskDraft.maskDataUrl,
        orderedInputImages[0].dataUrl,
      );
      if (coverage === 'full' && !options.allowFullMask) {
        setConfirmDialog({
          title: '确认编辑整张图片？',
          message:
            '当前遮罩覆盖了整张图片，提交后可能会重绘全部内容。是否继续？',
          confirmText: '继续提交',
          tone: 'warning',
          action: () => {
            void submitTask({ allowFullMask: true });
          },
        });
        return;
      }
      maskImageId = await storeImage(maskDraft.maskDataUrl, 'mask');
      cacheImage(maskImageId, maskDraft.maskDataUrl);
      maskTargetImageId = maskDraft.targetImageId;
    } catch (err) {
      if (!inputImages.some((img) => img.id === maskDraft.targetImageId)) {
        useStore.getState().clearMaskDraft();
      }
      showToast(err instanceof Error ? err.message : String(err), 'error');
      return;
    }
  }

  // 持久化输入图片到 IndexedDB（此前只在内存缓存中）
  for (const img of orderedInputImages) {
    await storeImage(img.dataUrl);
  }

  const paramsWithReferenceSize = await withReferenceImageSize(
    params,
    orderedInputImages,
  );
  const normalizedParams = normalizeParamsForSystemImageModel(
    normalizeParamsForSettings(paramsWithReferenceSize, requestSettings, {
      hasInputImages: orderedInputImages.length > 0,
    }),
    selectedSystemImageModel,
  );
  const normalizedParamPatch = getChangedParams(params, normalizedParams);
  if (Object.keys(normalizedParamPatch).length) {
    useStore.getState().setParams(normalizedParamPatch);
  }

  const taskId = genId();
  const task: TaskRecord = {
    id: taskId,
    prompt: prompt.trim(),
    params: normalizedParams,
    apiProvider:
      appMode === 'gallery'
        ? selectedSystemImageModel?.provider_name || 'system'
        : activeProfile.provider,
    apiProfileId: appMode === 'gallery' ? '' : activeProfile.id,
    apiProfileName:
      appMode === 'gallery'
        ? selectedSystemImageModel?.display_name ||
          selectedSystemImageModel?.site_model_id ||
          '系统图片模型'
        : activeProfile.name,
    apiMode:
      appMode === 'gallery'
        ? selectedSystemImageModel?.api_mode || 'images'
        : activeProfile.apiMode,
    apiModel:
      appMode === 'gallery'
        ? selectedSystemImageModel?.site_model_id || ''
        : activeProfile.model,
    streamPartialImages: normalizeStreamPartialImages(
      requestSettings.streamPartialImages,
    ),
    inputImageIds: orderedInputImages.map((i) => i.id),
    maskTargetImageId,
    maskImageId,
    outputImages: [],
    status: 'running',
    error: null,
    createdAt: Date.now(),
    finishedAt: null,
    elapsed: null,
  };

  const latestTasks = useStore.getState().tasks;
  useStore.getState().setTasks([task, ...latestTasks]);
  await putTask(task);
  useStore.getState().showToast('任务已提交', 'success');

  if (settings.clearInputAfterSubmit) {
    useStore.getState().setPrompt('');
    useStore.getState().clearInputImages();
  }
  useStore.getState().setReusedTaskApiProfile(null);

  // 异步调用 API
  executeTask(taskId);
}

function getActiveAgentConversation(): AgentConversation {
  const state = useStore.getState();
  const existing = state.agentConversations.find(
    (conversation) => conversation.id === state.activeAgentConversationId,
  );
  if (existing) return existing;

  const id = state.createAgentConversation();
  return useStore
    .getState()
    .agentConversations.find((conversation) => conversation.id === id)!;
}

function updateAgentConversation(
  conversationId: string,
  updater: (conversation: AgentConversation) => AgentConversation,
) {
  useStore.setState((state) => ({
    agentConversations: state.agentConversations.map((conversation) =>
      conversation.id === conversationId ? updater(conversation) : conversation,
    ),
  }));
}

function getAgentRoundControllerKey(conversationId: string, roundId: string) {
  return `${conversationId}:${roundId}`;
}

function createAgentAbortError() {
  return new DOMException('Agent 请求已停止', 'AbortError');
}

function appendAgentStoppedMessage(content: string) {
  const trimmed = content.trimEnd();
  if (!trimmed) return AGENT_STOPPED_MESSAGE;
  if (trimmed.endsWith(AGENT_STOPPED_MESSAGE)) return trimmed;
  return `${trimmed}\n\n${AGENT_STOPPED_MESSAGE}`;
}

function markAgentRoundTasksStopped(
  conversationId: string,
  roundId: string,
  now = Date.now(),
) {
  const runningTasks = useStore
    .getState()
    .tasks.filter(
      (task) =>
        task.status === 'running' &&
        task.agentConversationId === conversationId &&
        task.agentRoundId === roundId,
    );

  for (const task of runningTasks) {
    updateTaskInStore(task.id, {
      status: 'error',
      error: AGENT_STOPPED_MESSAGE,
      falRecoverable: false,
      customRecoverable: false,
      finishedAt: now,
      elapsed: Math.max(0, now - task.createdAt),
    });
  }
  return runningTasks.length > 0;
}

function markAgentRoundStopped(conversationId: string, roundId: string) {
  const now = Date.now();
  const stoppedTasks = markAgentRoundTasksStopped(conversationId, roundId, now);
  let stoppedRound = false;
  updateAgentConversation(conversationId, (current) => {
    const round = current.rounds.find((item) => item.id === roundId);
    if (!round || round.status !== 'running') return current;

    stoppedRound = true;
    const existingAssistantMessage = current.messages.find(
      (message) => message.roundId === roundId && message.role === 'assistant',
    );
    const assistantMessageId = existingAssistantMessage?.id ?? genId();
    return {
      ...current,
      updatedAt: now,
      rounds: current.rounds.map((item) =>
        item.id === roundId
          ? {
              ...item,
              ...(assistantMessageId ? { assistantMessageId } : {}),
              status: 'error',
              error: AGENT_STOPPED_MESSAGE,
              finishedAt: now,
            }
          : item,
      ),
      messages: existingAssistantMessage
        ? current.messages.map((message) =>
            message.id === existingAssistantMessage.id
              ? {
                  ...message,
                  content: appendAgentStoppedMessage(message.content),
                }
              : message,
          )
        : [
            ...current.messages,
            {
              id: assistantMessageId,
              role: 'assistant',
              content: AGENT_STOPPED_MESSAGE,
              roundId,
              createdAt: now,
            },
          ],
    };
  });
  return stoppedRound || stoppedTasks;
}

function appendAgentAssistantMessageContent(
  conversationId: string,
  messageId: string,
  delta: string,
) {
  if (!delta) return;
  updateAgentConversation(conversationId, (current) => ({
    ...current,
    updatedAt: Date.now(),
    messages: current.messages.map((message) =>
      message.id === messageId
        ? { ...message, content: `${message.content}${delta}` }
        : message,
    ),
  }));
}

async function generateAgentConversationTitle(
  conversationId: string,
  prompt: string,
  inputImageIds: string[],
  requestSettings: AppSettings,
  agentImageModelId: string,
  fallbackTitle: string,
) {
  useStore.setState((state) => {
    const next = {
      ...state.agentGeneratingTitleIds,
      [conversationId]: true as const,
    };
    return { agentGeneratingTitleIds: next };
  });
  try {
    const imageDataUrls = await readAgentImageDataUrls(inputImageIds);
    const title = await callAgentConversationTitleApi({
      settings: requestSettings,
      model: agentImageModelId,
      prompt,
      imageDataUrls,
    });
    if (!title || title === fallbackTitle) return;

    updateAgentConversation(conversationId, (current) => {
      const firstRound = current.rounds[0];
      if (
        !firstRound ||
        firstRound.prompt !== prompt ||
        current.title !== fallbackTitle
      )
        return current;
      return { ...current, title, updatedAt: Date.now() };
    });
  } catch {
    // Title generation is best-effort; keep the local fallback title on failure.
  } finally {
    useStore.setState((state) => {
      const next = { ...state.agentGeneratingTitleIds };
      delete next[conversationId];
      return { agentGeneratingTitleIds: next };
    });
  }
}

export function stopAgentResponse(
  conversationId = useStore.getState().activeAgentConversationId,
) {
  if (!conversationId) return;
  const conversation = useStore
    .getState()
    .agentConversations.find((item) => item.id === conversationId);
  if (!conversation) return;
  const activeRunningRound = [...getActiveAgentRounds(conversation)]
    .reverse()
    .find((round) => round.status === 'running');
  const runningRound =
    activeRunningRound ??
    conversation.rounds.find((round) => round.status === 'running');
  if (!runningRound) return;

  const controller = agentRoundControllers.get(
    getAgentRoundControllerKey(conversationId, runningRound.id),
  );
  if (controller) {
    controller.abort();
    if (markAgentRoundStopped(conversationId, runningRound.id)) {
      useStore.getState().showToast('已停止生成', 'info');
    }
    return;
  }

  markAgentRoundStopped(conversationId, runningRound.id);
  useStore.getState().showToast('已停止生成', 'info');
}

function getAgentRoundChildren(
  conversation: AgentConversation,
  parentRoundId: string | null,
) {
  return conversation.rounds.filter(
    (round) => (round.parentRoundId ?? null) === parentRoundId,
  );
}

function getLatestAgentLeafId(
  conversation: AgentConversation,
  startRoundId: string | null = null,
): string | null {
  let currentId = startRoundId;
  if (!currentId) {
    const roots = getAgentRoundChildren(conversation, null);
    currentId = roots[roots.length - 1]?.id ?? null;
  }

  while (currentId) {
    const children = getAgentRoundChildren(conversation, currentId);
    const nextId = children[children.length - 1]?.id ?? null;
    if (!nextId) return currentId;
    currentId = nextId;
  }

  return null;
}

export function getAgentRoundPath(
  conversation: AgentConversation,
  roundId: string | null,
): AgentRound[] {
  if (!roundId) return [];
  const byId = new Map(conversation.rounds.map((round) => [round.id, round]));
  const path: AgentRound[] = [];
  const seen = new Set<string>();
  let current = byId.get(roundId) ?? null;

  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    path.unshift(current);
    current = current.parentRoundId
      ? (byId.get(current.parentRoundId) ?? null)
      : null;
  }

  return path;
}

export function getActiveAgentRounds(
  conversation: AgentConversation,
): AgentRound[] {
  const activeRoundId =
    conversation.activeRoundId &&
    conversation.rounds.some((round) => round.id === conversation.activeRoundId)
      ? conversation.activeRoundId
      : getLatestAgentLeafId(conversation);
  return getAgentRoundPath(conversation, activeRoundId ?? null);
}

function reindexAgentRounds(
  conversation: AgentConversation,
): AgentConversation {
  const indexById = new Map<string, number>();
  const visit = (parentRoundId: string | null, depth: number) => {
    for (const child of getAgentRoundChildren(conversation, parentRoundId)) {
      indexById.set(child.id, depth);
      visit(child.id, depth + 1);
    }
  };
  visit(null, 1);
  return {
    ...conversation,
    rounds: conversation.rounds.map((round) => ({
      ...round,
      index: indexById.get(round.id) ?? round.index,
    })),
  };
}

export function remapAgentRoundMentionsForPathChange(
  content: string,
  oldPath: AgentRound[],
  newPath: AgentRound[],
) {
  if (!content || oldPath.length === 0) return content;
  const newIndexByRoundId = new Map(
    newPath.map((round, index) => [round.id, index + 1]),
  );
  return content.replace(
    AGENT_ROUND_IMAGE_MENTION_RE,
    (match, roundNumber: string, imageNumber: string) => {
      const oldRound = oldPath[Number(roundNumber) - 1];
      if (!oldRound) return match;
      const newRoundIndex = newIndexByRoundId.get(oldRound.id);
      if (!newRoundIndex) return `@已删除轮次图${imageNumber}`;
      return `@第${newRoundIndex}轮图${imageNumber}`;
    },
  );
}

export function deleteAgentRoundFromConversation(
  conversation: AgentConversation,
  roundId: string,
  now = Date.now(),
): AgentConversation {
  const targetRound = conversation.rounds.find((round) => round.id === roundId);
  if (!targetRound) return conversation;

  const oldPathByRoundId = new Map(
    conversation.rounds.map((round) => [
      round.id,
      getAgentRoundPath(conversation, round.id),
    ]),
  );
  const rounds = conversation.rounds
    .filter((candidate) => candidate.id !== roundId)
    .map((candidate) =>
      candidate.parentRoundId === roundId
        ? { ...candidate, parentRoundId: targetRound.parentRoundId ?? null }
        : candidate,
    );
  const messages = conversation.messages.filter(
    (candidate) => candidate.roundId !== roundId,
  );
  const nextConversation = reindexAgentRounds({
    ...conversation,
    rounds,
    messages,
    activeRoundId:
      conversation.activeRoundId === roundId
        ? null
        : (conversation.activeRoundId ?? null),
  });
  const newPathByRoundId = new Map(
    nextConversation.rounds.map((round) => [
      round.id,
      getAgentRoundPath(nextConversation, round.id),
    ]),
  );
  const remappedMessages = nextConversation.messages.map((message) => {
    if (!message.roundId) return message;
    const oldPath = oldPathByRoundId.get(message.roundId) ?? [];
    const newPath = newPathByRoundId.get(message.roundId) ?? [];
    const content = remapAgentRoundMentionsForPathChange(
      message.content,
      oldPath,
      newPath,
    );
    return content === message.content ? message : { ...message, content };
  });
  const withRemappedMessages = {
    ...nextConversation,
    messages: remappedMessages,
  };
  const activeRounds = getActiveAgentRounds(withRemappedMessages);
  return {
    ...withRemappedMessages,
    activeRoundId:
      withRemappedMessages.activeRoundId ??
      activeRounds[activeRounds.length - 1]?.id ??
      null,
    updatedAt: now,
  };
}

export function getAgentSiblingRounds(
  conversation: AgentConversation,
  round: AgentRound,
) {
  return getAgentRoundChildren(conversation, round.parentRoundId ?? null);
}

export function getAgentBranchLeafId(
  conversation: AgentConversation,
  roundId: string,
) {
  return getLatestAgentLeafId(conversation, roundId) ?? roundId;
}

function uniqueIds(ids: string[]) {
  return Array.from(new Set(ids.filter(Boolean)));
}

function addAgentReferencedImageIds(
  target: Set<string>,
  conversations = useStore.getState().agentConversations,
  inputDrafts = useStore.getState().agentInputDrafts,
) {
  for (const conversation of conversations) {
    for (const round of conversation.rounds) {
      for (const id of round.inputImageIds) target.add(id);
      if (round.maskImageId) target.add(round.maskImageId);
    }
    for (const message of conversation.messages) {
      if (message.maskImageId) target.add(message.maskImageId);
    }
  }
  for (const draft of Object.values(inputDrafts)) {
    for (const img of draft.inputImages) target.add(img.id);
  }
}

function addInputDraftReferencedImageIds(
  target: Set<string>,
  draft: AgentInputDraft | null,
) {
  if (!draft) return;
  for (const img of draft.inputImages) target.add(img.id);
}

function addTaskReferencedImageIds(target: Set<string>, task: TaskRecord) {
  for (const id of task.inputImageIds || []) target.add(id);
  if (task.maskImageId) target.add(task.maskImageId);
  for (const id of task.outputImages || []) target.add(id);
  for (const id of task.streamPartialImageIds || []) target.add(id);
}

async function deleteUnreferencedImageIds(imageIds: Iterable<string>) {
  const candidates = Array.from(new Set(Array.from(imageIds).filter(Boolean)));
  if (candidates.length === 0) return;

  const { tasks, inputImages, galleryInputDraft } = useStore.getState();
  const stillUsed = new Set<string>();
  for (const task of tasks) addTaskReferencedImageIds(stillUsed, task);
  addAgentReferencedImageIds(stillUsed);
  addInputDraftReferencedImageIds(stillUsed, galleryInputDraft);
  for (const img of inputImages) stillUsed.add(img.id);

  for (const imgId of candidates) {
    if (stillUsed.has(imgId)) continue;
    await deleteImage(imgId);
    imageCache.delete(imgId);
    thumbnailCache.delete(imgId);
  }
}

async function persistTaskStreamPartialImage(
  taskId: string,
  dataUrl: string,
  requestIndex = 0,
) {
  try {
    const imgId = await storeImage(dataUrl, 'generated');
    cacheImage(imgId, dataUrl);

    const latestTask = useStore
      .getState()
      .tasks.find((task) => task.id === taskId);
    if (!latestTask || latestTask.status === 'done') {
      await deleteUnreferencedImageIds([imgId]);
      return;
    }

    const currentIds = latestTask.streamPartialImageIds || [];
    if (currentIds.includes(imgId)) return;

    const index = Math.max(0, Math.trunc(requestIndex));
    const nextIds = [...currentIds];
    const previousId = nextIds[index];
    nextIds[index] = imgId;

    updateTaskInStore(taskId, {
      streamPartialImageIds: nextIds,
    });

    if (previousId && previousId !== imgId) {
      await deleteUnreferencedImageIds([previousId]);
    }
  } catch (err) {
    console.error(err);
  }
}

export function resolveSystemImageUrl(url: string) {
  if (!url || url.startsWith('blob:') || url.startsWith('data:')) return url;

  const appendAssetToken = (assetUrl: string) => {
    const token = Storage.get(LOGGED_TOKEN_STORAGE_KEY);
    if (!token) return assetUrl;
    try {
      const parsed = new URL(assetUrl, window.location.origin);
      parsed.searchParams.set('Authorization', token);
      return parsed.pathname + parsed.search + parsed.hash;
    } catch {
      return assetUrl;
    }
  };

  try {
    const parsed = new URL(url, window.location.origin);
    const match = parsed.pathname.match(
      /^\/uploads\/ai-images\/([^/]+)\/([^/?#]+)$/,
    );
    if (match) {
      return appendAssetToken(
        `/answer/api/v1/ai-image/assets/${encodeURIComponent(match[1])}/${encodeURIComponent(match[2])}`,
      );
    }
  } catch {
    // Keep the original URL if parsing fails.
  }

  if (/^https?:\/\//i.test(url)) return url;
  const apiBase = process.env.REACT_APP_API_URL || '';
  if (apiBase && apiBase !== '/' && url.startsWith('/')) {
    return `${apiBase.replace(/\/$/, '')}${url}`;
  }
  return url;
}

async function imageUrlToDataUrl(url: string) {
  const resolvedUrl = resolveSystemImageUrl(url);
  const token = Storage.get(LOGGED_TOKEN_STORAGE_KEY) || '';
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = token;
  const response = await fetch(resolvedUrl, {
    credentials: 'include',
    headers,
  });
  if (!response.ok) throw new Error(`图片读取失败：${response.status}`);
  const blob = await response.blob();
  if (!blob.type.startsWith('image/'))
    throw new Error('接口返回的不是有效图片');
  return blobToDataUrl(blob);
}

async function imageUrlToDataUrlWithRetry(url: string) {
  let lastError: unknown;
  for (
    let attempt = 0;
    attempt <= SYSTEM_IMAGE_ASSET_RETRY_DELAYS_MS.length;
    attempt += 1
  ) {
    try {
      return await imageUrlToDataUrl(url);
    } catch (err) {
      lastError = err;
      const delay = SYSTEM_IMAGE_ASSET_RETRY_DELAYS_MS[attempt];
      if (delay == null) break;
      await new Promise((resolve) => {
        window.setTimeout(resolve, delay);
      });
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function cacheTaskImageUrlsInBackground(
  taskId: string,
  imageUrls: string[],
  startIndex = 0,
) {
  if (!imageUrls.length) return;
  const resolvedOutputImages: string[] = [];
  for (let index = 0; index < imageUrls.length; index += 1) {
    const imageUrl = imageUrls[index];
    try {
      const dataUrl = await imageUrlToDataUrlWithRetry(imageUrl);
      const imgId = await storeImage(dataUrl, 'generated');
      cacheImage(imgId, dataUrl);
      resolvedOutputImages[index] = imgId;
    } catch (err) {
      console.warn('Failed to cache generated image URL', err);
    }
  }
  if (!resolvedOutputImages.some(Boolean)) return;

  const latestTask = useStore
    .getState()
    .tasks.find((item) => item.id === taskId);
  if (!latestTask) return;

  const nextOutputImages = [...(latestTask.outputImages || [])];
  for (let index = 0; index < resolvedOutputImages.length; index += 1) {
    const imgId = resolvedOutputImages[index];
    if (!imgId) continue;
    const outputIndex = startIndex + index;
    if (nextOutputImages[outputIndex]) continue;
    nextOutputImages[outputIndex] = imgId;
  }
  if (
    nextOutputImages.join('\n') === (latestTask.outputImages || []).join('\n')
  )
    return;

  updateTaskInStore(taskId, {
    outputImages: nextOutputImages,
  });
}

interface SystemImageStreamResult {
  generationId: string;
  size: string;
  images: string[];
  imageUrls: string[];
  expiresAt: number;
}

function parseSystemImageStreamData(
  block: string,
): Record<string, unknown> | null {
  const data = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).replace(/^ /, ''))
    .join('\n')
    .trim();
  if (!data || data === '[DONE]') return null;
  const parsed = JSON.parse(data) as unknown;
  return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : null;
}

function getStreamString(source: Record<string, unknown>, key: string) {
  const value = source[key];
  return typeof value === 'string' ? value : '';
}

function getStreamNumber(
  source: Record<string, unknown>,
  key: string,
): number | undefined {
  const value = source[key];
  return typeof value === 'number' && Number.isFinite(value)
    ? value
    : undefined;
}

function collectResponsesStreamImages(
  payload: Record<string, unknown>,
  images: string[],
) {
  const collectItem = (item: unknown) => {
    if (!item || typeof item !== 'object') return;
    const record = item as Record<string, unknown>;
    if (getStreamString(record, 'type') !== 'image_generation_call') return;
    const result = getStreamString(record, 'result');
    if (result) images.push(normalizeBase64Image(result, 'image/png'));
  };

  const item = payload.item;
  collectItem(item);

  const response = payload.response;
  if (response && typeof response === 'object') {
    const output = (response as Record<string, unknown>).output;
    if (Array.isArray(output)) output.forEach(collectItem);
  }

  const output = payload.output;
  if (Array.isArray(output)) output.forEach(collectItem);
}

async function generateAiImageStream(
  params: AiImageGenerateParams,
  onPartialImage: (partial: {
    image: string;
    partialImageIndex?: number;
    itemIndex?: number;
  }) => void,
  onGenerationCreated?: (generationId: string) => void,
): Promise<SystemImageStreamResult> {
  const token = Storage.get(LOGGED_TOKEN_STORAGE_KEY) || '';
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  };
  if (token) headers.Authorization = token;

  const response = await fetch('/answer/api/v1/ai-image/generations', {
    method: 'POST',
    credentials: 'include',
    headers,
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    let message = `流式生成请求失败：${response.status}`;
    try {
      const payload = await response.clone().json();
      const msg = typeof payload?.msg === 'string' ? payload.msg.trim() : '';
      if (msg) message = msg;
    } catch {
      // keep the status fallback when the error payload is not JSON
    }
    throw new Error(message);
  }
  if (!response.body) throw new Error('接口未返回可读取的流式响应');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const finalImages: string[] = [];
  const finalImageUrls: string[] = [];
  let generationId = '';
  let responseSize = '';
  let expiresAt = 0;

  const processBlock = (block: string) => {
    const event = parseSystemImageStreamData(block);
    if (!event) return;
    const error = event.error;
    if (error && typeof error === 'object' && 'message' in error) {
      throw new Error(
        String((error as { message?: unknown }).message || '流式请求失败'),
      );
    }
    const type = getStreamString(event, 'type');
    if (type === 'ping' || type === 'keep-alive' || type === 'heartbeat') {
      return;
    }
    if (type === 'image_generation.created') {
      generationId = getStreamString(event, 'generation_id') || generationId;
      if (generationId) onGenerationCreated?.(generationId);
      responseSize = responseSize || getStreamString(event, 'size');
      expiresAt = getStreamNumber(event, 'expires_at') ?? expiresAt;
      return;
    }
    if (
      type === 'image_generation.partial_image' ||
      type === 'image_edit.partial_image' ||
      type === 'response.image_generation_call.partial_image'
    ) {
      const b64 =
        getStreamString(event, 'b64_json') ||
        getStreamString(event, 'partial_image_b64');
      if (b64) {
        onPartialImage({
          image: normalizeBase64Image(b64, 'image/png'),
          partialImageIndex: getStreamNumber(event, 'partial_image_index'),
          itemIndex: getStreamNumber(event, 'item_index'),
        });
      }
      return;
    }

    if (
      type === 'image_generation.completed' ||
      type === 'image_edit.completed'
    ) {
      const b64 = getStreamString(event, 'b64_json');
      const imageUrl = getStreamString(event, 'url');
      if (b64) finalImages.push(normalizeBase64Image(b64, 'image/png'));
      if (imageUrl) finalImageUrls.push(imageUrl);
      responseSize = responseSize || getStreamString(event, 'size');
      return;
    }

    if (type === 'response.output_item.done' || type === 'response.completed') {
      collectResponsesStreamImages(event, finalImages);
      return;
    }

    if (getStreamString(event, 'object') === 'image.generation.result') {
      const data = Array.isArray(event.data) ? event.data : [];
      data.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const record = item as Record<string, unknown>;
        const b64 = getStreamString(record, 'b64_json');
        const imageUrl = getStreamString(record, 'url');
        if (b64) finalImages.push(normalizeBase64Image(b64, 'image/png'));
        if (imageUrl) finalImageUrls.push(imageUrl);
      });
      generationId = getStreamString(event, 'generation_id') || generationId;
      responseSize = getStreamString(event, 'size') || responseSize;
      expiresAt = getStreamNumber(event, 'expires_at') ?? expiresAt;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let separatorIndex = buffer.search(/\r?\n\r?\n/);
    while (separatorIndex >= 0) {
      const separator = buffer.match(/\r?\n\r?\n/)?.[0] ?? '\n\n';
      const block = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + separator.length);
      processBlock(block);
      separatorIndex = buffer.search(/\r?\n\r?\n/);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) processBlock(buffer);
  if (finalImages.length === 0 && finalImageUrls.length === 0) {
    throw new Error('流式接口未返回最终图片');
  }
  return {
    generationId,
    size: responseSize,
    images: Array.from(new Set(finalImages)),
    imageUrls: Array.from(new Set(finalImageUrls)),
    expiresAt,
  };
}

function normalizeSystemGenerationStatus(status?: string): TaskStatus {
  if (status === 'failed' || status === 'error') return 'error';
  if (
    status === 'queued' ||
    status === 'saving' ||
    status === 'generating' ||
    status === 'processing' ||
    status === 'in_progress'
  )
    return 'running';
  return 'done';
}

function normalizeSystemOutputFormat(
  outputFormat?: string,
): TaskParams['output_format'] {
  if (
    outputFormat === 'png' ||
    outputFormat === 'jpeg' ||
    outputFormat === 'webp'
  )
    return outputFormat;
  return DEFAULT_PARAMS.output_format;
}

function normalizeSystemQuality(quality?: string): TaskParams['quality'] {
  if (
    quality === 'auto' ||
    quality === 'low' ||
    quality === 'medium' ||
    quality === 'high'
  )
    return quality;
  return DEFAULT_PARAMS.quality;
}

function normalizeSystemModeration(
  moderation?: string,
): TaskParams['moderation'] {
  return moderation === 'low' ? 'low' : 'auto';
}

async function mapSystemImageGenerationToTask(
  generation: AiImageGeneration,
): Promise<TaskRecord> {
  const rawImageUrls = (generation.image_urls || []).filter(Boolean);
  const streamPartialImageUrls = generation.partial_image_urls || [];

  const status = normalizeSystemGenerationStatus(generation.status);
  const createdAt = (generation.created_at || 0) * 1000 || Date.now();
  const finishedAt =
    status === 'running' ? null : (generation.updated_at || 0) * 1000 || null;
  const params: TaskParams = {
    size: generation.size || DEFAULT_PARAMS.size,
    quality: normalizeSystemQuality(generation.quality),
    output_format: normalizeSystemOutputFormat(generation.output_format),
    output_compression:
      typeof generation.output_compression === 'number' &&
      generation.output_compression > 0
        ? generation.output_compression
        : null,
    moderation: normalizeSystemModeration(generation.moderation),
    n: Math.max(1, generation.count || rawImageUrls.length || 1),
  };

  return {
    id: generation.generation_id,
    prompt: generation.prompt || '',
    params,
    apiProvider: generation.provider_name || 'system',
    apiProfileId: '',
    apiProfileName:
      generation.provider_name ||
      generation.site_model_id ||
      generation.provider_model_id ||
      '系统图片模型',
    apiMode: generation.api_mode || 'images',
    apiModel: generation.site_model_id,
    inputImageIds: [],
    maskTargetImageId: null,
    maskImageId: null,
    outputImages: [],
    streamPartialImageUrls: streamPartialImageUrls.length
      ? streamPartialImageUrls
      : undefined,
    rawImageUrls: rawImageUrls.length ? rawImageUrls : undefined,
    systemGenerationId: generation.generation_id,
    rawResponsePayload: generation.response_output || undefined,
    actualParams: params,
    status,
    error: generation.error || null,
    createdAt,
    finishedAt,
    elapsed: finishedAt ? Math.max(0, finishedAt - createdAt) : null,
    falRecoverable: false,
    customRecoverable: false,
    sourceMode: 'gallery',
  };
}

async function getSelectedSystemImageModel(): Promise<AiImageModel> {
  const state = useStore.getState();
  if (!state.systemImageModels.length) {
    await state.loadSystemImageModels();
  }

  const latestState = useStore.getState();
  const selectedModel =
    latestState.systemImageModels.find(
      (model) => model.site_model_id === latestState.selectedSystemImageModelId,
    ) || latestState.systemImageModels[0];
  if (!selectedModel) {
    throw new Error(
      latestState.systemImageModelsError ||
        '暂无可用图片模型，请联系管理员检查配置',
    );
  }
  return selectedModel;
}

async function getSelectedAgentSystemImageModel(): Promise<AiImageModel> {
  const state = useStore.getState();
  if (!state.systemImageModelsLoaded && !state.systemImageModelsLoading) {
    await state.loadSystemImageModels();
  } else if (!state.systemImageModels.length) {
    await state.loadSystemImageModels();
  }

  const latestState = useStore.getState();
  const agentModel = getPreferredImageAgentThinkingModel(
    latestState.systemImageModels,
    latestState.selectedSystemImageModelId,
  );
  if (!agentModel) {
    throw new Error(getAgentThinkingModelUnavailableMessage(latestState));
  }
  if (agentModel.site_model_id !== latestState.selectedSystemImageModelId) {
    useStore.setState({
      selectedSystemImageModelId: agentModel.site_model_id,
      params: normalizeParamsForSystemImageModel(
        latestState.params,
        agentModel,
      ),
    });
  }
  return agentModel;
}

async function generateSystemAgentImage(opts: {
  taskId: string;
  batchItemId: string;
  prompt: string;
  params: TaskParams;
  referenceImageDataUrls: string[];
  maskDataUrl?: string;
  signal: AbortSignal;
}): Promise<BatchImageCallResult> {
  if (opts.signal.aborted) throw createAgentAbortError();
  try {
    const selectedModel = await getSelectedSystemImageModel();
    if (opts.signal.aborted) throw createAgentAbortError();
    const normalizedParams = normalizeParamsForSystemImageModel(
      await withReferenceDataUrlSize(opts.params, opts.referenceImageDataUrls),
      selectedModel,
    );

    const response = await generateAiImage({
      prompt: replaceImageMentionsForApi(
        opts.prompt,
        opts.referenceImageDataUrls.length,
      ),
      model: selectedModel.site_model_id,
      size: normalizedParams.size,
      quality: normalizedParams.quality,
      output_format: normalizedParams.output_format,
      ...(opts.params.output_compression == null
        ? {}
        : { output_compression: opts.params.output_compression }),
      moderation: normalizedParams.moderation,
      count: 1,
      reference_images: opts.referenceImageDataUrls,
      ...(opts.maskDataUrl ? { mask_image: opts.maskDataUrl } : {}),
    });
    if (opts.signal.aborted) throw createAgentAbortError();

    const imageUrl = response.image_urls?.[0];
    if (!imageUrl) {
      throw new Error('接口没有返回图片');
    }

    const dataUrl = await imageUrlToDataUrl(imageUrl);
    if (opts.signal.aborted) throw createAgentAbortError();

    const image: AgentApiResultImage = {
      toolCallId: opts.taskId,
      action: opts.referenceImageDataUrls.length > 0 ? 'auto' : 'generate',
      dataUrl,
      actualParams: {
        ...normalizedParams,
        size: response.size || normalizedParams.size,
        n: 1,
      },
      revisedPrompt: opts.prompt,
    };

    return {
      batchItemId: opts.batchItemId,
      image,
      error: null,
      rawResponsePayload: JSON.stringify(response, null, 2),
    };
  } catch (err) {
    if (opts.signal.aborted) throw createAgentAbortError();
    return {
      batchItemId: opts.batchItemId,
      image: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function persistAgentGeneratedImageToBackend(opts: {
  imageDataUrl: string;
  prompt: string;
  params: Partial<TaskParams>;
  model: string;
}) {
  if (!opts.model || !opts.imageDataUrl) return null;
  try {
    return await saveAiImageAgentGeneration({
      prompt: opts.prompt.trim() || 'Agent image',
      model: opts.model,
      size: opts.params.size || DEFAULT_PARAMS.size,
      quality: opts.params.quality || DEFAULT_PARAMS.quality,
      output_format: opts.params.output_format || DEFAULT_PARAMS.output_format,
      ...(opts.params.output_compression == null
        ? {}
        : { output_compression: opts.params.output_compression }),
      moderation: opts.params.moderation || DEFAULT_PARAMS.moderation,
      images: [opts.imageDataUrl],
    });
  } catch (err) {
    console.error('Failed to persist agent image generation', err);
    return null;
  }
}

async function executeSystemImageTask(taskId: string) {
  const task = useStore.getState().tasks.find((item) => item.id === taskId);
  if (!task) return;

  try {
    const inputDataUrls: string[] = [];
    for (const imgId of task.inputImageIds) {
      const dataUrl = await ensureImageCached(imgId);
      if (!dataUrl) throw new Error('输入图片已不存在');
      inputDataUrls.push(dataUrl);
    }

    const maskDataUrl = task.maskImageId
      ? await ensureImageCached(task.maskImageId)
      : '';
    if (task.maskImageId && !maskDataUrl) throw new Error('遮罩图片已不存在');

    const count = Math.max(1, Math.trunc(task.params.n || 1));
    const systemModel = useStore
      .getState()
      .systemImageModels.find((model) => model.site_model_id === task.apiModel);
    const shouldStreamSystemImage =
      systemModel?.supports_stream === true &&
      inputDataUrls.length === 0 &&
      !maskDataUrl;
    const requestParams: AiImageGenerateParams = {
      prompt: replaceImageMentionsForApi(task.prompt, inputDataUrls.length),
      model:
        task.apiModel ||
        getActiveApiProfile(useStore.getState().settings).model,
      size: task.params.size,
      quality: task.params.quality,
      output_format: task.params.output_format,
      ...(task.params.output_compression == null
        ? {}
        : { output_compression: task.params.output_compression }),
      moderation: task.params.moderation,
      count,
      reference_images: inputDataUrls,
      ...(maskDataUrl ? { mask_image: maskDataUrl } : {}),
    };
    const response = shouldStreamSystemImage
      ? await generateAiImageStream(
          {
            ...requestParams,
            stream: true,
            partial_images: normalizeStreamPartialImages(
              task.streamPartialImages,
              DEFAULT_SETTINGS.streamPartialImages,
            ),
          },
          (partial) => {
            useStore
              .getState()
              .setTaskStreamPreview(taskId, partial.image, partial.itemIndex);
            void persistTaskStreamPartialImage(
              taskId,
              partial.image,
              partial.itemIndex,
            );
          },
          (generationId) => {
            updateTaskInStore(taskId, { systemGenerationId: generationId });
          },
        )
      : await generateAiImage(requestParams);

    const latestBeforeSuccess = useStore
      .getState()
      .tasks.find((item) => item.id === taskId);
    if (!latestBeforeSuccess || latestBeforeSuccess.status !== 'running')
      return;

    const responseImageUrls =
      'imageUrls' in response ? response.imageUrls : response.image_urls || [];
    const responseImages = 'images' in response ? response.images || [] : [];
    const responseGenerationId =
      'generationId' in response
        ? response.generationId
        : response.generation_id;

    const outputIds: string[] = [];
    for (const image of responseImages) {
      const imgId = await storeImage(image, 'generated');
      cacheImage(imgId, image);
      outputIds.push(imgId);
    }
    const pendingImageUrls = responseImageUrls.slice(outputIds.length);
    if (outputIds.length === 0 && responseImageUrls.length === 0)
      throw new Error('接口没有返回图片');

    const latestBeforeUpdate = useStore
      .getState()
      .tasks.find((item) => item.id === taskId);
    if (!latestBeforeUpdate || latestBeforeUpdate.status !== 'running') return;
    const partialImageIdsToClean =
      latestBeforeUpdate.streamPartialImageIds || [];

    updateTaskInStore(taskId, {
      outputImages: outputIds,
      systemGenerationId: responseGenerationId,
      streamPartialImageIds: undefined,
      rawImageUrls: responseImageUrls.length ? responseImageUrls : undefined,
      actualParams: {
        ...task.params,
        size: response.size || task.params.size,
        n: Math.max(outputIds.length, responseImageUrls.length),
      },
      status: 'done',
      error: null,
      finishedAt: Date.now(),
      elapsed: Date.now() - task.createdAt,
      falRecoverable: false,
      customRecoverable: false,
    });

    void cacheTaskImageUrlsInBackground(
      taskId,
      pendingImageUrls,
      outputIds.length,
    );
    useStore.getState().setTaskStreamPreview(taskId);
    void deleteUnreferencedImageIds(partialImageIdsToClean);

    useStore
      .getState()
      .showToast(
        `生成完成，共 ${Math.max(outputIds.length, responseImageUrls.length)} 张图片`,
        'success',
      );
    showTaskCompletionNotification(
      '图像生成完成',
      `生成完成，共 ${Math.max(outputIds.length, responseImageUrls.length)} 张图片。`,
    );
    window.dispatchEvent(new CustomEvent('hcai-subscription-updated'));
    void useStore.getState().loadSystemImageGenerations({ silent: true });
  } catch (err) {
    useStore.getState().setTaskStreamPreview(taskId);
    const latestTask =
      useStore.getState().tasks.find((item) => item.id === taskId) ?? task;
    if (latestTask.status !== 'running') return;

    updateTaskInStore(taskId, {
      status: 'error',
      error: getTaskErrorMessage(err),
      ...getRawErrorPayload(err),
      falRecoverable: false,
      customRecoverable: false,
      finishedAt: Date.now(),
      elapsed: Date.now() - task.createdAt,
    });
    useStore.getState().setDetailTaskId(taskId);
  } finally {
    for (const imgId of task.inputImageIds) imageCache.delete(imgId);
  }
}

async function readAgentImageDataUrls(ids: string[]) {
  const dataUrls: string[] = [];
  for (const id of ids) {
    const dataUrl = await ensureImageCached(id);
    if (dataUrl) dataUrls.push(dataUrl);
  }
  return dataUrls;
}

async function createAgentUserInputItem(
  conversation: AgentConversation,
  round: AgentRound,
  message: AgentMessage,
  tasks: TaskRecord[],
) {
  const imageDataUrls = await readAgentImageDataUrls(round.inputImageIds);
  const rounds = getAgentRoundPath(conversation, round.id);
  const text = replaceAgentPromptImageReferencesForApi(
    message.content,
    round,
    rounds,
    tasks,
  );
  const referenceText =
    round.inputImageIds.length > 0
      ? `\n\n<available_refs>${round.inputImageIds.map((_, index) => `\n  <ref id="${getAgentCurrentReferenceId(round, index)}" />`).join('')}\n</available_refs>`
      : '';
  return {
    role: 'user',
    content: [
      { type: 'input_text', text: `${text}${referenceText}` },
      ...imageDataUrls.map((dataUrl) => ({
        type: 'input_image',
        image_url: dataUrl,
      })),
    ],
  };
}

async function createAgentGeneratedImagesInputItem(
  round: AgentRound,
  tasks: TaskRecord[],
) {
  const contentParts: Array<{
    type: string;
    text?: string;
    image_url?: string;
  }> = [];
  let imageIndex = 0;
  for (const taskId of round.outputTaskIds) {
    const task = tasks.find((item) => item.id === taskId);
    if (!task) {
      contentParts.push({
        type: 'input_text',
        text: `<removed_ref id="${getAgentGeneratedImageReferenceId(round, imageIndex)}" />`,
      });
      imageIndex += 1;
      continue;
    }
    for (const imageId of task.outputImages) {
      const dataUrl = await ensureImageCached(imageId);
      if (dataUrl) {
        contentParts.push({ type: 'input_image', image_url: dataUrl });
      }
      const refId = getAgentGeneratedImageReferenceId(round, imageIndex);
      const prompt = truncateAgentReferencePrompt(task.prompt || '');
      const promptAttribute = prompt
        ? ` prompt="${escapeXmlAttribute(prompt)}"`
        : '';
      contentParts.push({
        type: 'input_text',
        text: `<ref id="${refId}"${promptAttribute} />`,
      });
      imageIndex += 1;
    }
  }
  if (contentParts.length === 0) return null;
  return { role: 'user', content: contentParts };
}

async function createAgentBatchImagesInputItem(
  round: AgentRound,
  tasks: TaskRecord[],
  batchTaskIds: string[],
) {
  const contentParts: Array<{
    type: string;
    text?: string;
    image_url?: string;
  }> = [];
  // Count existing images in the round to compute correct imageIndex offset
  let baseImageIndex = 0;
  for (const taskId of round.outputTaskIds) {
    if (batchTaskIds.includes(taskId)) break;
    const task = tasks.find((item) => item.id === taskId);
    baseImageIndex += task ? task.outputImages.length : 1;
  }
  let imageIndex = baseImageIndex;
  for (const taskId of batchTaskIds) {
    const task = tasks.find((item) => item.id === taskId);
    if (!task || task.status !== 'done') continue;
    for (const imgId of task.outputImages) {
      const dataUrl = await ensureImageCached(imgId);
      if (dataUrl) {
        contentParts.push({ type: 'input_image', image_url: dataUrl });
      }
      const refId = getAgentGeneratedImageReferenceId(round, imageIndex);
      const prompt = truncateAgentReferencePrompt(task.prompt || '');
      const promptAttribute = prompt
        ? ` prompt="${escapeXmlAttribute(prompt)}"`
        : '';
      contentParts.push({
        type: 'input_text',
        text: `<ref id="${refId}"${promptAttribute} />`,
      });
      imageIndex += 1;
    }
  }
  if (contentParts.length === 0) return null;
  return { role: 'user', content: contentParts };
}

function escapeXmlAttribute(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function truncateAgentReferencePrompt(prompt: string) {
  const normalized = prompt.replace(/\s+/g, ' ').trim();
  return normalized.length > 1200
    ? `${normalized.slice(0, 1200)}...`
    : normalized;
}

function createAgentAssistantFallbackItem(text: string) {
  return {
    role: 'assistant',
    content: [{ type: 'output_text', text }],
  };
}

function parseResponseOutputFromPayload(
  rawResponsePayload?: string,
): ResponsesOutputItem[] | null {
  if (!rawResponsePayload) return null;
  try {
    const payload = JSON.parse(rawResponsePayload) as { output?: unknown };
    return Array.isArray(payload.output)
      ? (payload.output as ResponsesOutputItem[])
      : null;
  } catch {
    return null;
  }
}

function sanitizeResponseOutputItemForInput(
  item: ResponsesOutputItem,
): unknown | null {
  if (item.type === 'web_search_call') return null;
  if (item.type === 'image_generation_call') return null;

  if (item.type === 'message') {
    const content = (item.content ?? [])
      .map((part) => {
        if (typeof part.text !== 'string') return null;
        if (part.type === 'output_text' || part.type === 'text') {
          return { type: 'output_text', text: part.text };
        }
        return null;
      })
      .filter((part): part is { type: 'output_text'; text: string } =>
        Boolean(part),
      );

    return content.length > 0 ? { role: 'assistant', content } : null;
  }

  return item;
}

function filterAgentRoundResponseOutputForInput(
  _round: AgentRound,
  _tasks: TaskRecord[],
  output: ResponsesOutputItem[],
) {
  // image_generation_call items are now dropped by sanitizeResponseOutputItemForInput;
  // this filter is kept as a structural pass-through for future use.
  return output;
}

function scrubResponseOutputForDeletedAgentTasks(
  round: AgentRound,
  output: ResponsesOutputItem[],
  deletedTasks: TaskRecord[],
) {
  const deletedTaskIds = new Set(deletedTasks.map((task) => task.id));
  const deletedToolCallIds = new Set(
    deletedTasks
      .filter((task) => task.agentRoundId === round.id && task.agentToolCallId)
      .map((task) => task.agentToolCallId!),
  );
  if (deletedTaskIds.size === 0) return output;

  let anonymousImageIndex = 0;
  return output.filter((item) => {
    if (item.type !== 'image_generation_call') return true;

    if (typeof item.id === 'string' && item.id) {
      return !deletedToolCallIds.has(item.id);
    }

    const taskId = round.outputTaskIds[anonymousImageIndex];
    anonymousImageIndex += 1;
    return !deletedTaskIds.has(taskId);
  });
}

function scrubAgentConversationsForDeletedTasks(
  conversations: AgentConversation[],
  deletedTasks: TaskRecord[],
) {
  if (deletedTasks.length === 0) return conversations;

  return conversations.map((conversation) => ({
    ...conversation,
    rounds: conversation.rounds.map((round) => {
      const roundDeletedTasks = deletedTasks.filter((task) =>
        round.outputTaskIds.includes(task.id),
      );
      if (roundDeletedTasks.length === 0 || !round.responseOutput?.length)
        return round;
      return {
        ...round,
        responseOutput: scrubResponseOutputForDeletedAgentTasks(
          round,
          round.responseOutput,
          roundDeletedTasks,
        ),
      };
    }),
  }));
}

function scrubTaskRawResponsePayloadForDeletedTasks(
  task: TaskRecord,
  conversations: AgentConversation[],
  deletedTasks: TaskRecord[],
) {
  if (!task.rawResponsePayload || !task.agentRoundId) return task;

  const round = conversations
    .flatMap((conversation) => conversation.rounds)
    .find((item) => item.id === task.agentRoundId);
  if (!round) return task;

  const roundDeletedTasks = deletedTasks.filter((item) =>
    round.outputTaskIds.includes(item.id),
  );
  if (roundDeletedTasks.length === 0) return task;

  try {
    const payload = JSON.parse(task.rawResponsePayload) as ResponsesApiResponse;
    if (!Array.isArray(payload.output)) return task;
    const output = scrubResponseOutputForDeletedAgentTasks(
      round,
      payload.output,
      roundDeletedTasks,
    );
    if (output.length === payload.output.length) return task;
    return {
      ...task,
      rawResponsePayload: JSON.stringify({ ...payload, output }, null, 2),
    };
  } catch {
    return task;
  }
}

async function scrubAgentOutputPayloadsForDeletedTasks(
  deletedTasks: TaskRecord[],
  remainingTasks: TaskRecord[],
) {
  if (deletedTasks.length === 0) return remainingTasks;

  const conversations = scrubAgentConversationsForDeletedTasks(
    useStore.getState().agentConversations,
    deletedTasks,
  );
  const scrubbedTasks = remainingTasks.map((task) =>
    scrubTaskRawResponsePayloadForDeletedTasks(
      task,
      conversations,
      deletedTasks,
    ),
  );
  useStore.setState({ agentConversations: conversations });

  for (const task of scrubbedTasks) {
    const previous = remainingTasks.find((item) => item.id === task.id);
    if (previous?.rawResponsePayload !== task.rawResponsePayload)
      await putTask(task);
  }

  return scrubbedTasks;
}

function sanitizeResponseOutputForInput(
  output: ResponsesOutputItem[],
  options: { allowPendingFunctionCalls?: boolean } = {},
) {
  const items = output
    .map(sanitizeResponseOutputItemForInput)
    .filter((item): item is unknown => item != null);
  if (options.allowPendingFunctionCalls) return items;

  const functionCallIds = new Set<string>();
  const functionOutputCallIds = new Set<string>();
  for (const item of items) {
    if (!isRecord(item)) continue;
    const callId = typeof item.call_id === 'string' ? item.call_id : '';
    if (!callId) continue;
    if (item.type === 'function_call') functionCallIds.add(callId);
    if (item.type === 'function_call_output') functionOutputCallIds.add(callId);
  }

  return items.filter((item) => {
    if (!isRecord(item)) return true;
    const callId = typeof item.call_id === 'string' ? item.call_id : '';
    if (item.type === 'function_call')
      return callId && functionOutputCallIds.has(callId);
    if (item.type === 'function_call_output')
      return callId && functionCallIds.has(callId);
    return true;
  });
}

function mergeResponseOutputItems(
  previous: ResponsesOutputItem[],
  next: ResponsesOutputItem[],
) {
  const merged = [...previous];
  for (const item of next) {
    const index = item.id
      ? merged.findIndex((existing) => existing.id === item.id)
      : -1;
    if (index >= 0) merged[index] = item;
    else merged.push(item);
  }
  return merged;
}

function createAgentContinuationInputItem(
  newImageRefs: string[],
  toolCallsUsed: number,
  maxToolCalls: number,
) {
  const lines = [
    '[System] The app has saved your generated outputs and is continuing the same Agent turn.',
  ];
  if (newImageRefs.length > 0) {
    lines.push(
      `The following image ref ids are now available for you to reference in subsequent generate_image prompts: ${newImageRefs.join(', ')}`,
    );
  }
  lines.push(
    'Continue generating. Do NOT repeat what you already said in earlier responses.',
    'If you still need another round after this (e.g. more dependent images), call continue_generation.',
    `Tool-call budget: ${toolCallsUsed}/${maxToolCalls} used.`,
  );
  return {
    role: 'user',
    content: [
      {
        type: 'input_text',
        text: lines.join('\n'),
      },
    ],
  };
}

function buildAgentContinuationInput(
  baseInput: unknown[],
  round: AgentRound,
  tasks: TaskRecord[],
  currentRoundOutput: ResponsesOutputItem[],
  toolCallsUsed: number,
  maxToolCalls: number,
) {
  const input = [
    ...baseInput,
    ...sanitizeResponseOutputForInput(currentRoundOutput, {
      allowPendingFunctionCalls: true,
    }),
  ];
  const newImageRefs = collectAgentRoundOutputImageSlots(round, tasks)
    .map((imageId, index) =>
      imageId
        ? `<ref id="${getAgentGeneratedImageReferenceId(round, index)}" />`
        : null,
    )
    .filter((ref): ref is string => Boolean(ref));
  input.push(
    createAgentContinuationInputItem(newImageRefs, toolCallsUsed, maxToolCalls),
  );
  return input;
}

function getAgentRoundResponseOutput(
  round: AgentRound,
  tasks: TaskRecord[],
): ResponsesOutputItem[] | null {
  if (round.responseOutput?.length) return round.responseOutput;

  for (const taskId of round.outputTaskIds) {
    const task = tasks.find((item) => item.id === taskId);
    const output = parseResponseOutputFromPayload(task?.rawResponsePayload);
    if (output?.length) return output;
  }

  return null;
}

async function buildAgentApiInput(
  conversation: AgentConversation,
  currentRound: AgentRound,
  tasks: TaskRecord[],
): Promise<unknown[]> {
  const input: unknown[] = [];
  const rounds = getAgentRoundPath(conversation, currentRound.id);

  for (const round of rounds) {
    const userMessage = conversation.messages.find(
      (message) => message.id === round.userMessageId,
    );
    if (!userMessage) continue;

    input.push(
      await createAgentUserInputItem(conversation, round, userMessage, tasks),
    );
    if (round.id === currentRound.id) continue;

    const output = getAgentRoundResponseOutput(round, tasks);
    if (output?.length) {
      const sanitizedOutput = sanitizeResponseOutputForInput(
        filterAgentRoundResponseOutputForInput(round, tasks, output),
      );
      if (sanitizedOutput.length > 0) {
        input.push(...sanitizedOutput);
      } else {
        // All output items were filtered (e.g. only image_generation_call); add fallback
        const assistantMessage = round.assistantMessageId
          ? conversation.messages.find(
              (message) => message.id === round.assistantMessageId,
            )
          : null;
        input.push(
          createAgentAssistantFallbackItem(
            assistantMessage?.content || '图像已生成。',
          ),
        );
      }
    } else {
      const assistantMessage = round.assistantMessageId
        ? conversation.messages.find(
            (message) => message.id === round.assistantMessageId,
          )
        : null;
      input.push(
        createAgentAssistantFallbackItem(
          assistantMessage?.content || '[No text response]',
        ),
      );
    }

    // Inject generated images as a separate user message with input_image parts
    if (round.outputTaskIds.length > 0) {
      const imagesItem = await createAgentGeneratedImagesInputItem(
        round,
        tasks,
      );
      if (imagesItem) input.push(imagesItem);
    }
  }

  return input;
}

export async function submitAgentMessage() {
  const state = useStore.getState();
  const { settings, prompt, inputImages, maskDraft, params, showToast } = state;

  const normalizedSettings = normalizeSettings(settings);
  const requestSettings = normalizedSettings;
  let selectedSystemImageModel: AiImageModel;
  try {
    selectedSystemImageModel = await getSelectedAgentSystemImageModel();
  } catch (err) {
    showToast(err instanceof Error ? err.message : String(err), 'error');
    if (!state.systemImageModelsLoading) void state.loadSystemImageModels();
    return;
  }

  const trimmedPrompt = prompt.trim();
  if (!trimmedPrompt) {
    showToast('请输入消息', 'error');
    return;
  }

  const conversation = getActiveAgentConversation();
  if (conversation.rounds.some((round) => round.status === 'running')) {
    showToast('请等待生成完成，或先停止生成', 'info');
    return;
  }

  let orderedInputImages = inputImages;
  let maskImageId: string | null = null;
  let maskTargetImageId: string | null = null;

  if (maskDraft) {
    try {
      orderedInputImages = orderInputImagesForMask(
        inputImages,
        maskDraft.targetImageId,
      );
      await validateMaskMatchesImage(
        maskDraft.maskDataUrl,
        orderedInputImages[0].dataUrl,
      );
      maskImageId = await storeImage(maskDraft.maskDataUrl, 'mask');
      cacheImage(maskImageId, maskDraft.maskDataUrl);
      maskTargetImageId = maskDraft.targetImageId;
    } catch (err) {
      if (!inputImages.some((img) => img.id === maskDraft.targetImageId)) {
        state.clearMaskDraft();
      }
      showToast(err instanceof Error ? err.message : String(err), 'error');
      return;
    }
  }

  const inputImageIds = uniqueIds(orderedInputImages.map((image) => image.id));

  for (const image of orderedInputImages) {
    await storeImage(image.dataUrl);
  }

  const now = Date.now();
  const editingRound = state.agentEditingRoundId
    ? (conversation.rounds.find(
        (item) => item.id === state.agentEditingRoundId,
      ) ?? null)
    : null;
  const editingRoundAssistantMessage = editingRound?.assistantMessageId
    ? (conversation.messages.find(
        (message) => message.id === editingRound.assistantMessageId,
      ) ?? null)
    : (conversation.messages.find(
        (message) =>
          message.roundId === editingRound?.id && message.role === 'assistant',
      ) ?? null);
  const editingRoundHasAssistantMessage = Boolean(editingRoundAssistantMessage);
  const editingRoundHasErrorAssistantMessage = Boolean(
    editingRound?.status === 'error' &&
      editingRoundAssistantMessage?.content.startsWith('请求失败：'),
  );
  const editingRoundHasChildren = editingRound
    ? conversation.rounds.some(
        (round) => (round.parentRoundId ?? null) === editingRound.id,
      )
    : false;
  const shouldAppendToEditingRound = Boolean(
    editingRound &&
      !editingRoundHasChildren &&
      (!editingRoundHasAssistantMessage ||
        editingRoundHasErrorAssistantMessage),
  );
  const roundId =
    shouldAppendToEditingRound && editingRound ? editingRound.id : genId();
  const userMessageId =
    shouldAppendToEditingRound && editingRound
      ? editingRound.userMessageId
      : genId();
  const activeRounds = getActiveAgentRounds(conversation);
  const activeLeafId = activeRounds[activeRounds.length - 1]?.id ?? null;
  const parentRoundId = editingRound
    ? (editingRound.parentRoundId ?? null)
    : activeLeafId;
  const parentPath = parentRoundId
    ? getAgentRoundPath(conversation, parentRoundId)
    : [];
  const paramsWithReferenceSize = await withReferenceImageSize(
    params,
    orderedInputImages,
  );
  const normalizedParams = {
    ...normalizeParamsForSystemImageModel(
      normalizeParamsForSettings(paramsWithReferenceSize, requestSettings, {
        hasInputImages: inputImageIds.length > 0,
      }),
      selectedSystemImageModel,
    ),
    n: DEFAULT_PARAMS.n,
  };
  const round: AgentRound = {
    id: roundId,
    index:
      shouldAppendToEditingRound && editingRound
        ? editingRound.index
        : parentPath.length + 1,
    parentRoundId,
    ...(editingRoundHasErrorAssistantMessage && editingRoundAssistantMessage
      ? { assistantMessageId: editingRoundAssistantMessage.id }
      : {}),
    userMessageId,
    prompt: trimmedPrompt,
    inputImageIds,
    maskTargetImageId,
    maskImageId,
    outputTaskIds: [],
    status: 'running',
    error: null,
    createdAt: now,
    finishedAt: null,
  };
  const userMessage: AgentMessage = {
    id: userMessageId,
    role: 'user',
    content: trimmedPrompt,
    roundId,
    inputImageIds,
    maskTargetImageId,
    maskImageId,
    createdAt: now,
  };

  let fallbackTitle: string | null = null;
  updateAgentConversation(conversation.id, (current) => {
    const nextTitle =
      current.rounds.length === 0
        ? createAgentConversationTitle(trimmedPrompt, current.title)
        : current.title;
    if (current.rounds.length === 0) fallbackTitle = nextTitle;
    const messages = shouldAppendToEditingRound
      ? current.messages.some((message) => message.id === userMessageId)
        ? current.messages.map((message) => {
            if (message.id === userMessageId) return userMessage;
            if (
              editingRoundHasErrorAssistantMessage &&
              message.id === editingRoundAssistantMessage?.id
            ) {
              return { ...message, content: '', outputTaskIds: [] };
            }
            return message;
          })
        : [...current.messages, userMessage]
      : [...current.messages, userMessage];

    return {
      ...current,
      title: nextTitle,
      activeRoundId: roundId,
      updatedAt: now,
      rounds: shouldAppendToEditingRound
        ? current.rounds.map((item) => (item.id === roundId ? round : item))
        : [...current.rounds, round],
      messages,
    };
  });

  state.setPrompt('');
  state.clearInputImages();
  state.clearMaskDraft();
  state.setAgentEditingRoundId(null);

  if (fallbackTitle) {
    void generateAgentConversationTitle(
      conversation.id,
      trimmedPrompt,
      inputImageIds,
      requestSettings,
      selectedSystemImageModel.site_model_id,
      fallbackTitle,
    );
  }

  void executeAgentRound(
    conversation.id,
    roundId,
    normalizedParams,
    requestSettings,
    selectedSystemImageModel.site_model_id,
  );
}

export async function regenerateAgentAssistantMessage(
  conversationId: string,
  roundId: string,
) {
  const state = useStore.getState();
  const { settings, params, showToast } = state;

  const normalizedSettings = normalizeSettings(settings);
  const requestSettings = normalizedSettings;
  let selectedSystemImageModel: AiImageModel;
  try {
    selectedSystemImageModel = await getSelectedAgentSystemImageModel();
  } catch (err) {
    showToast(err instanceof Error ? err.message : String(err), 'error');
    if (!state.systemImageModelsLoading) void state.loadSystemImageModels();
    return;
  }

  const conversation = state.agentConversations.find(
    (item) => item.id === conversationId,
  );
  const sourceRound =
    conversation?.rounds.find((item) => item.id === roundId) ?? null;
  const sourceUserMessage = sourceRound
    ? (conversation?.messages.find(
        (message) => message.id === sourceRound.userMessageId,
      ) ?? null)
    : null;
  if (!conversation || !sourceRound || !sourceUserMessage) {
    showToast('找不到要重新生成的 Agent 消息', 'error');
    return;
  }

  if (conversation.rounds.some((round) => round.status === 'running')) {
    showToast('请等待生成完成，或先停止生成', 'info');
    return;
  }

  const inputImageIds = uniqueIds(sourceRound.inputImageIds);
  const sourceInputImages: InputImage[] = [];
  for (const imageId of inputImageIds) {
    const dataUrl = await ensureImageCached(imageId);
    if (dataUrl) sourceInputImages.push({ id: imageId, dataUrl });
  }
  const paramsWithReferenceSize = await withReferenceImageSize(
    params,
    sourceInputImages,
  );
  const normalizedParams = {
    ...normalizeParamsForSystemImageModel(
      normalizeParamsForSettings(paramsWithReferenceSize, requestSettings, {
        hasInputImages: inputImageIds.length > 0,
      }),
      selectedSystemImageModel,
    ),
    n: DEFAULT_PARAMS.n,
  };
  const now = Date.now();
  if (sourceRound.status === 'error') {
    const assistantMessageId =
      sourceRound.assistantMessageId ??
      conversation.messages.find(
        (message) =>
          message.roundId === sourceRound.id && message.role === 'assistant',
      )?.id;
    updateAgentConversation(conversationId, (current) => ({
      ...current,
      activeRoundId: sourceRound.id,
      updatedAt: now,
      rounds: current.rounds.map((round) =>
        round.id === sourceRound.id
          ? {
              ...round,
              outputTaskIds: [],
              responseId: undefined,
              responseOutput: undefined,
              status: 'running',
              error: null,
              finishedAt: null,
            }
          : round,
      ),
      messages: assistantMessageId
        ? current.messages.map((message) =>
            message.id === assistantMessageId
              ? { ...message, content: '', outputTaskIds: [] }
              : message,
          )
        : current.messages,
    }));
    state.setAgentEditingRoundId(null);
    void executeAgentRound(
      conversationId,
      sourceRound.id,
      normalizedParams,
      requestSettings,
      selectedSystemImageModel.site_model_id,
    );
    return;
  }

  const newRoundId = genId();
  const newUserMessageId = genId();
  const newRound: AgentRound = {
    id: newRoundId,
    index: sourceRound.index,
    parentRoundId: sourceRound.parentRoundId ?? null,
    userMessageId: newUserMessageId,
    prompt: sourceRound.prompt || sourceUserMessage.content.trim(),
    inputImageIds,
    maskTargetImageId:
      sourceRound.maskTargetImageId ??
      sourceUserMessage.maskTargetImageId ??
      null,
    maskImageId:
      sourceRound.maskImageId ?? sourceUserMessage.maskImageId ?? null,
    outputTaskIds: [],
    status: 'running',
    error: null,
    createdAt: now,
    finishedAt: null,
  };
  const newUserMessage: AgentMessage = {
    id: newUserMessageId,
    role: 'user',
    content: sourceUserMessage.content,
    roundId: newRoundId,
    inputImageIds,
    maskTargetImageId:
      sourceRound.maskTargetImageId ??
      sourceUserMessage.maskTargetImageId ??
      null,
    maskImageId:
      sourceRound.maskImageId ?? sourceUserMessage.maskImageId ?? null,
    createdAt: now,
  };

  updateAgentConversation(conversationId, (current) => ({
    ...current,
    activeRoundId: newRoundId,
    updatedAt: now,
    rounds: [...current.rounds, newRound],
    messages: [...current.messages, newUserMessage],
  }));
  state.setAgentEditingRoundId(null);
  void executeAgentRound(
    conversationId,
    newRoundId,
    normalizedParams,
    requestSettings,
    selectedSystemImageModel.site_model_id,
  );
}

async function executeAgentRound(
  conversationId: string,
  roundId: string,
  params: TaskParams,
  requestSettings: AppSettings,
  agentImageModelId: string,
) {
  const startedAt = Date.now();
  const controller = new AbortController();
  const controllerKey = getAgentRoundControllerKey(conversationId, roundId);
  agentRoundControllers.set(controllerKey, controller);
  try {
    const latestState = useStore.getState();
    const conversation = latestState.agentConversations.find(
      (item) => item.id === conversationId,
    );
    if (!conversation) return;
    const round = conversation.rounds.find((item) => item.id === roundId);
    const userMessage = round
      ? conversation.messages.find(
          (message) => message.id === round.userMessageId,
        )
      : null;
    if (!round || !userMessage) return;
    const agentSystemImageModel =
      latestState.systemImageModels.find(
        (model) => model.site_model_id === agentImageModelId,
      ) ?? (await getSelectedSystemImageModel());
    const maskDataUrl = round.maskImageId
      ? await ensureImageCached(round.maskImageId)
      : undefined;
    if (round.maskImageId && !maskDataUrl) throw new Error('遮罩图片已不存在');
    const shouldStreamAgentResponses =
      agentSystemImageModel.supports_stream === true;
    const streamPartialImages = normalizeStreamPartialImages(
      requestSettings.streamPartialImages,
    );

    const apiInput = await buildAgentApiInput(
      conversation,
      round,
      latestState.tasks,
    );
    if (controller.signal.aborted) throw createAgentAbortError();
    const existingAssistantMessage = round.assistantMessageId
      ? (conversation.messages.find(
          (message) => message.id === round.assistantMessageId,
        ) ?? null)
      : (conversation.messages.find(
          (message) =>
            message.roundId === roundId && message.role === 'assistant',
        ) ?? null);
    const assistantMessageId = existingAssistantMessage?.id ?? genId();
    const shouldStreamAssistantMessage = true;
    const streamingTaskIds: string[] = [];
    const taskIdByToolCallId = new Map<string, string>();

    const attachTaskToAgentRound = (taskId: string) => {
      if (streamingTaskIds.includes(taskId)) return;
      streamingTaskIds.push(taskId);
      updateAgentConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        rounds: current.rounds.map((item) =>
          item.id === roundId
            ? {
                ...item,
                outputTaskIds: item.outputTaskIds.includes(taskId)
                  ? item.outputTaskIds
                  : [...item.outputTaskIds, taskId],
              }
            : item,
        ),
        messages: current.messages.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                outputTaskIds: [
                  ...new Set([...(message.outputTaskIds ?? []), taskId]),
                ],
              }
            : message,
        ),
      }));
    };

    const ensureStreamingAgentTask = async (
      toolCallId: string,
      taskPrompt = '',
      inputImageIds = round.inputImageIds ?? [],
      options: {
        createdAt?: number;
        agentBatchCallId?: string;
        maskTargetImageId?: string | null;
        maskImageId?: string | null;
        systemImageModel?: AiImageModel | null;
      } = {},
    ) => {
      const existingTaskId = taskIdByToolCallId.get(toolCallId);
      if (existingTaskId) return existingTaskId;

      const existingTask = useStore
        .getState()
        .tasks.find((task) => task.agentToolCallId === toolCallId);
      if (existingTask) {
        taskIdByToolCallId.set(toolCallId, existingTask.id);
        attachTaskToAgentRound(existingTask.id);
        return existingTask.id;
      }

      const task: TaskRecord = {
        id: genId(),
        prompt: taskPrompt,
        params: { ...params, n: 1 },
        apiProvider:
          options.systemImageModel?.provider_name ||
          agentSystemImageModel.provider_name ||
          'system',
        apiProfileId: '',
        apiProfileName:
          options.systemImageModel?.display_name ||
          options.systemImageModel?.site_model_id ||
          agentSystemImageModel.display_name ||
          agentSystemImageModel.site_model_id ||
          '系统图片模型',
        apiMode:
          options.systemImageModel?.api_mode || agentSystemImageModel.api_mode,
        apiModel:
          options.systemImageModel?.site_model_id ||
          agentSystemImageModel.site_model_id,
        inputImageIds,
        streamPartialImages,
        maskTargetImageId:
          options.maskTargetImageId !== undefined
            ? options.maskTargetImageId
            : (round.maskTargetImageId ?? null),
        maskImageId:
          options.maskImageId !== undefined
            ? options.maskImageId
            : (round.maskImageId ?? null),
        outputImages: [],
        status: 'running',
        error: null,
        createdAt: options.createdAt ?? Date.now(),
        finishedAt: null,
        elapsed: null,
        sourceMode: 'agent',
        agentConversationId: conversationId,
        agentRoundId: roundId,
        agentMessageId: assistantMessageId,
        agentToolCallId: toolCallId,
        ...(options.agentBatchCallId
          ? { agentBatchCallId: options.agentBatchCallId }
          : {}),
      };

      taskIdByToolCallId.set(toolCallId, task.id);
      useStore.getState().setTasks([task, ...useStore.getState().tasks]);
      attachTaskToAgentRound(task.id);
      await putTask(task);
      return task.id;
    };

    const completeAgentImageTask = async (
      image: AgentApiResultImage,
      rawResponsePayload?: string,
    ) => {
      const toolCallId = image.toolCallId ?? genId();
      const taskId = await ensureStreamingAgentTask(toolCallId);
      const latestTask = useStore
        .getState()
        .tasks.find((task) => task.id === taskId);
      if (latestTask?.status === 'done' && latestTask.outputImages.length > 0) {
        if (rawResponsePayload && !latestTask.rawResponsePayload) {
          updateTaskInStore(taskId, { rawResponsePayload });
        }
        return taskId;
      }

      const imgId = await storeImage(image.dataUrl, 'generated');
      cacheImage(imgId, image.dataUrl);
      const actualParams: Partial<TaskParams> = {
        ...(Object.keys(image.actualParams ?? {}).length
          ? image.actualParams
          : {}),
        n: 1,
      };
      const prompt = image.revisedPrompt ?? latestTask?.prompt ?? '';
      const persisted = await persistAgentGeneratedImageToBackend({
        imageDataUrl: image.dataUrl,
        prompt,
        params: actualParams,
        model: latestTask?.apiModel || agentSystemImageModel.site_model_id,
      });
      updateTaskInStore(taskId, {
        prompt,
        outputImages: [imgId],
        actualParams,
        actualParamsByImage: { [imgId]: actualParams },
        revisedPromptByImage: image.revisedPrompt
          ? { [imgId]: image.revisedPrompt }
          : undefined,
        systemGenerationId: persisted?.generation_id,
        rawImageUrls: persisted?.image_urls?.length
          ? persisted.image_urls
          : undefined,
        rawResponsePayload,
        status: 'done',
        error: null,
        finishedAt: Date.now(),
        elapsed: Date.now() - (latestTask?.createdAt ?? startedAt),
        agentToolAction: image.action,
      });
      useStore.getState().setTaskStreamPreview(taskId);
      return taskId;
    };

    if (shouldStreamAssistantMessage) {
      updateAgentConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        rounds: current.rounds.map((item) =>
          item.id === roundId ? { ...item, assistantMessageId } : item,
        ),
        messages: current.messages.some(
          (message) => message.id === assistantMessageId,
        )
          ? current.messages.map((message) =>
              message.id === assistantMessageId
                ? { ...message, content: '', outputTaskIds: [] }
                : message,
            )
          : [
              ...current.messages,
              {
                id: assistantMessageId,
                role: 'assistant',
                content: '',
                roundId,
                createdAt: Date.now(),
              },
            ],
      }));
    }
    const maxToolCalls = Number.isFinite(requestSettings.agentMaxToolRounds)
      ? Math.max(1, Math.trunc(requestSettings.agentMaxToolRounds))
      : DEFAULT_AGENT_MAX_TOOL_ROUNDS;
    let apiInputForTurn = apiInput;
    let accumulatedOutputItems: ResponsesOutputItem[] = [];
    let accumulatedText = '';
    const textSegments: string[] = [];
    let lastResponseId: string | undefined;
    let toolCallsUsed = 0;
    let reachedToolLimit = false;
    let pendingToolTextSeparator = false;

    // Helper: resolve reference image ids to data URLs for batch image calls
    const resolveReferenceImages = async (
      referenceIds: string[],
    ): Promise<{ dataUrls: string[]; imageIds: string[] }> => {
      const dataUrls: string[] = [];
      const imageIds: string[] = [];
      for (const refId of referenceIds) {
        // Resolve both generated image refs and current/user input refs from XML tags.
        const latestConv = useStore
          .getState()
          .agentConversations.find((item) => item.id === conversationId);
        if (!latestConv) continue;
        for (const r of getAgentRoundPath(latestConv, roundId)) {
          for (let imgIdx = 0; imgIdx < r.inputImageIds.length; imgIdx++) {
            const currentRefId = getAgentCurrentReferenceId(r, imgIdx);
            if (currentRefId === refId) {
              const imageId = r.inputImageIds[imgIdx];
              const dataUrl = await ensureImageCached(imageId);
              if (dataUrl) dataUrls.push(dataUrl);
              imageIds.push(imageId);
            }
          }
          const outputImages = collectAgentRoundOutputImageSlots(
            r,
            useStore.getState().tasks,
          );
          for (let imgIdx = 0; imgIdx < outputImages.length; imgIdx++) {
            const generatedRefId = getAgentGeneratedImageReferenceId(r, imgIdx);
            if (generatedRefId === refId) {
              const imageId = outputImages[imgIdx];
              if (!imageId) continue;
              const dataUrl = await ensureImageCached(imageId);
              if (dataUrl) dataUrls.push(dataUrl);
              imageIds.push(imageId);
            }
          }
        }
      }
      return { dataUrls, imageIds };
    };

    // Helper: execute a generate_image_batch function call concurrently
    const executeSingleFunctionCall = async (
      functionCallItem: ResponsesOutputItem,
    ): Promise<string> => {
      const callId = functionCallItem.call_id ?? functionCallItem.id ?? genId();
      const args = functionCallItem.arguments ?? '';
      const parsed = parseSingleImageCallArguments(args);

      if (!parsed) {
        return JSON.stringify({ error: 'Invalid or empty image arguments' });
      }

      const selectedSystemImageModel = await getSelectedSystemImageModel();
      const referenceIds = uniqueIds(extractAgentReferenceIds(parsed.prompt));
      const references = await resolveReferenceImages(referenceIds);
      const taskId = await ensureStreamingAgentTask(
        callId,
        parsed.prompt,
        references.imageIds,
        {
          createdAt: Date.now(),
          maskTargetImageId: round.maskTargetImageId ?? null,
          maskImageId: round.maskImageId ?? null,
          systemImageModel: selectedSystemImageModel,
        },
      );

      const result = await generateSystemAgentImage({
        taskId: callId,
        params,
        batchItemId: callId,
        prompt: parsed.prompt,
        referenceImageDataUrls: references.dataUrls,
        maskDataUrl,
        signal: controller.signal,
      });

      if (result.image) {
        await completeAgentImageTask(
          { ...result.image, toolCallId: callId },
          result.rawResponsePayload,
        );
      } else {
        updateTaskInStore(taskId, {
          status: 'error',
          error: result.error || '图片生成失败',
          finishedAt: Date.now(),
          elapsed:
            Date.now() -
            (useStore.getState().tasks.find((task) => task.id === taskId)
              ?.createdAt ?? startedAt),
        });
      }

      toolCallsUsed += result.image ? 1 : 0;
      return JSON.stringify({
        id: callId,
        status: result.image ? 'done' : 'error',
        ...(result.error ? { error: result.error } : {}),
      });
    };

    const executeBatchFunctionCall = async (
      functionCallItem: ResponsesOutputItem,
    ): Promise<string> => {
      const callId = functionCallItem.call_id ?? '';
      const args = functionCallItem.arguments ?? '';
      const batchItems = parseBatchImageCallArguments(args);

      if (!batchItems || batchItems.length === 0) {
        return JSON.stringify({ error: 'Invalid or empty batch arguments' });
      }

      const selectedSystemImageModel = await getSelectedSystemImageModel();

      // Create task cards in model-provided order before starting network calls.
      const batchExecutionItems: Array<{
        item: { id: string; prompt: string };
        batchToolCallId: string;
        references: { dataUrls: string[]; imageIds: string[] };
        referenceIds: string[];
      }> = [];
      for (const item of batchItems) {
        const referenceIds = uniqueIds(extractAgentReferenceIds(item.prompt));
        const references = await resolveReferenceImages(referenceIds);
        const batchToolCallId = genId();
        await ensureStreamingAgentTask(
          batchToolCallId,
          item.prompt,
          references.imageIds,
          {
            createdAt: Date.now(),
            maskTargetImageId: null,
            maskImageId: null,
            systemImageModel: selectedSystemImageModel,
            ...(callId ? { agentBatchCallId: callId } : {}),
          },
        );
        batchExecutionItems.push({
          item,
          batchToolCallId,
          references,
          referenceIds,
        });
      }

      // Keep cards visible immediately, but avoid bursting many image_generation calls at once.
      const runBatchImageItem = async ({
        item,
        batchToolCallId,
        references,
        referenceIds,
      }: (typeof batchExecutionItems)[number]) => {
        const batchParams = normalizeParamsForSystemImageModel(
          await withReferenceDataUrlSize(params, references.dataUrls),
          selectedSystemImageModel,
        );
        const batchResult = await callBatchImageSingle({
          model: agentImageModelId,
          params: batchParams,
          batchItemId: item.id,
          prompt: item.prompt,
          referenceImageDataUrls: references.dataUrls,
          referenceIds,
          stream: shouldStreamAgentResponses,
          streamPartialImages: shouldStreamAgentResponses
            ? streamPartialImages
            : undefined,
          signal: controller.signal,
          onImageToolStarted: shouldStreamAssistantMessage
            ? async () => {
                if (controller.signal.aborted) return;
                await ensureStreamingAgentTask(batchToolCallId);
              }
            : undefined,
          onPartialImage: shouldStreamAssistantMessage
            ? async ({ image, partialImageIndex }) => {
                if (controller.signal.aborted) return;
                const taskId = await ensureStreamingAgentTask(batchToolCallId);
                if (controller.signal.aborted) return;
                useStore
                  .getState()
                  .setTaskStreamPreview(taskId, image, partialImageIndex);
                if (partialImageIndex === 0 || partialImageIndex == null) {
                  void persistTaskStreamPartialImage(taskId, image, 0);
                }
              }
            : undefined,
          onImageToolCompleted: shouldStreamAssistantMessage
            ? async (image) => {
                if (controller.signal.aborted) return;
                await completeAgentImageTask({
                  ...image,
                  toolCallId: batchToolCallId,
                });
              }
            : undefined,
        });

        if (batchResult.image) {
          await completeAgentImageTask(
            { ...batchResult.image, toolCallId: batchToolCallId },
            batchResult.rawResponsePayload,
          );
        }

        return batchResult;
      };

      const batchResults: PromiseSettledResult<BatchImageCallResult>[] = [];
      let nextBatchIndex = 0;
      const workerCount = Math.min(2, batchExecutionItems.length);
      const workers = Array.from({ length: workerCount }, async () => {
        while (nextBatchIndex < batchExecutionItems.length) {
          const index = nextBatchIndex;
          nextBatchIndex += 1;
          try {
            batchResults[index] = {
              status: 'fulfilled',
              value: await runBatchImageItem(batchExecutionItems[index]),
            };
          } catch (reason) {
            batchResults[index] = { status: 'rejected', reason };
          }
        }
      });
      await Promise.all(workers);

      // Build function_call_output
      const outputImages: Array<{
        id: string;
        status: string;
        error?: string;
      }> = [];
      for (let i = 0; i < batchItems.length; i++) {
        const settled = batchResults[i];
        const batchItem = batchItems[i];
        if (settled.status === 'fulfilled') {
          const r = settled.value;
          outputImages.push({
            id: r.batchItemId,
            status: r.image ? 'done' : 'error',
            ...(r.error ? { error: r.error } : {}),
          });
        } else {
          outputImages.push({
            id: batchItem.id,
            status: 'error',
            error:
              settled.reason instanceof Error
                ? settled.reason.message
                : String(settled.reason),
          });
        }
      }

      const successCount = outputImages.filter(
        (img) => img.status === 'done',
      ).length;
      toolCallsUsed += successCount;

      return JSON.stringify({ images: outputImages });
    };

    while (true) {
      if (controller.signal.aborted) throw createAgentAbortError();
      const textBeforeResponse = accumulatedText;
      let currentResponseOutputItems: ResponsesOutputItem[] = [];
      const result = await callAgentResponsesApi({
        settings: requestSettings,
        model: agentImageModelId,
        params,
        stream: shouldStreamAgentResponses,
        streamPartialImages:
          shouldStreamAgentResponses === true ? streamPartialImages : undefined,
        input: apiInputForTurn,
        maskDataUrl,
        signal: controller.signal,
        onTextDelta: shouldStreamAssistantMessage
          ? (delta) => {
              if (controller.signal.aborted) return;
              if (pendingToolTextSeparator && delta && accumulatedText.trim()) {
                accumulatedText += '\n\n';
                appendAgentAssistantMessageContent(
                  conversationId,
                  assistantMessageId,
                  '\n\n',
                );
              }
              pendingToolTextSeparator = false;
              accumulatedText += delta;
              appendAgentAssistantMessageContent(
                conversationId,
                assistantMessageId,
                delta,
              );
            }
          : undefined,
        onOutputItems: shouldStreamAssistantMessage
          ? (outputItems) => {
              if (controller.signal.aborted) return;
              currentResponseOutputItems = outputItems;
              updateAgentConversation(conversationId, (current) => ({
                ...current,
                rounds: current.rounds.map((item) =>
                  item.id === roundId
                    ? {
                        ...item,
                        responseOutput: mergeResponseOutputItems(
                          accumulatedOutputItems,
                          outputItems,
                        ),
                      }
                    : item,
                ),
              }));
            }
          : undefined,
        onImageToolStarted: shouldStreamAssistantMessage
          ? async ({ toolCallId }) => {
              if (controller.signal.aborted) return;
              await ensureStreamingAgentTask(toolCallId);
            }
          : undefined,
        onImagePartialImage: shouldStreamAssistantMessage
          ? async ({ toolCallId, image, partialImageIndex }) => {
              if (controller.signal.aborted) return;
              const taskId = await ensureStreamingAgentTask(toolCallId);
              if (controller.signal.aborted) return;
              useStore
                .getState()
                .setTaskStreamPreview(taskId, image, partialImageIndex);
              if (partialImageIndex === 0 || partialImageIndex == null) {
                void persistTaskStreamPartialImage(taskId, image, 0);
              }
            }
          : undefined,
        onImageToolCompleted: shouldStreamAssistantMessage
          ? async (image) => {
              if (controller.signal.aborted) return;
              await completeAgentImageTask(image);
            }
          : undefined,
      });
      if (controller.signal.aborted) throw createAgentAbortError();

      lastResponseId = result.responseId ?? lastResponseId;
      currentResponseOutputItems = currentResponseOutputItems.length
        ? currentResponseOutputItems
        : (result.outputItems ?? []);
      accumulatedOutputItems = mergeResponseOutputItems(
        accumulatedOutputItems,
        currentResponseOutputItems,
      );

      for (const image of result.images) {
        const completedTaskId = await completeAgentImageTask(
          image,
          result.rawResponsePayload,
        );
        const promptRefIds = uniqueIds(
          extractAgentReferenceIds(image.revisedPrompt ?? ''),
        );
        if (promptRefIds.length > 0) {
          const promptRefs = await resolveReferenceImages(promptRefIds);
          if (promptRefs.imageIds.length > 0) {
            const latestTask = useStore
              .getState()
              .tasks.find((item) => item.id === completedTaskId);
            if (latestTask) {
              const mergedInputIds = uniqueIds([
                ...latestTask.inputImageIds,
                ...promptRefs.imageIds,
              ]);
              if (mergedInputIds.length !== latestTask.inputImageIds.length) {
                updateTaskInStore(completedTaskId, {
                  inputImageIds: mergedInputIds,
                });
              }
            }
          }
        }
      }
      toolCallsUsed += result.images.length;

      const responseText = result.text.trim();
      if (responseText && accumulatedText === textBeforeResponse) {
        const textToAppend = accumulatedText
          ? `\n\n${responseText}`
          : responseText;
        accumulatedText += textToAppend;
        if (shouldStreamAssistantMessage)
          appendAgentAssistantMessageContent(
            conversationId,
            assistantMessageId,
            textToAppend,
          );
      }
      const newTextInThisResponse = accumulatedText
        .slice(textBeforeResponse.length)
        .trim();
      if (newTextInThisResponse) textSegments.push(newTextInThisResponse);

      if (result.rawResponsePayload && streamingTaskIds.length > 0) {
        for (const taskId of streamingTaskIds) {
          const latestTask = useStore
            .getState()
            .tasks.find((task) => task.id === taskId);
          if (latestTask && !latestTask.rawResponsePayload)
            updateTaskInStore(taskId, {
              rawResponsePayload: result.rawResponsePayload,
            });
        }
      }

      // Check for function calls that require continuation
      const singleFunctionCalls = currentResponseOutputItems.filter(
        (item) =>
          item.type === 'function_call' && item.name === 'generate_image',
      );
      const batchFunctionCalls = currentResponseOutputItems.filter(
        (item) =>
          item.type === 'function_call' && item.name === 'generate_image_batch',
      );
      const continueFunctionCalls = currentResponseOutputItems.filter(
        (item) =>
          item.type === 'function_call' && item.name === 'continue_generation',
      );

      // Collect function_call_output items for all function calls that need responses
      const functionCallOutputs: ResponsesOutputItem[] = [];

      if (singleFunctionCalls.length > 0) {
        for (const fc of singleFunctionCalls) {
          const output = await executeSingleFunctionCall(fc);
          functionCallOutputs.push({
            type: 'function_call_output',
            call_id: fc.call_id,
            output,
          });
        }
      }

      if (batchFunctionCalls.length > 0) {
        for (const fc of batchFunctionCalls) {
          const output = await executeBatchFunctionCall(fc);
          functionCallOutputs.push({
            type: 'function_call_output',
            call_id: fc.call_id,
            output,
          });
        }
      }

      for (const fc of continueFunctionCalls) {
        functionCallOutputs.push({
          type: 'function_call_output',
          call_id: fc.call_id,
          output: JSON.stringify({ status: 'continued' }),
        });
      }

      // If no function calls need output → model decided the task is done → break
      if (functionCallOutputs.length === 0) {
        updateAgentConversation(conversationId, (current) => ({
          ...current,
          updatedAt: Date.now(),
          rounds: current.rounds.map((item) =>
            item.id === roundId
              ? {
                  ...item,
                  responseId: lastResponseId,
                  responseOutput: accumulatedOutputItems,
                }
              : item,
          ),
        }));
        break;
      }

      const accumulatedOutputItemsWithFunctionOutputs =
        mergeResponseOutputItems(accumulatedOutputItems, functionCallOutputs);

      updateAgentConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        rounds: current.rounds.map((item) =>
          item.id === roundId
            ? {
                ...item,
                responseId: lastResponseId,
                responseOutput: accumulatedOutputItemsWithFunctionOutputs,
              }
            : item,
        ),
      }));

      if (toolCallsUsed >= maxToolCalls) {
        reachedToolLimit = true;
        break;
      }

      // Build continuation input with function call outputs and available refs
      const latestConversation = useStore
        .getState()
        .agentConversations.find((item) => item.id === conversationId);
      const latestRound = latestConversation?.rounds.find(
        (item) => item.id === roundId,
      );
      if (!latestRound) break;

      const continuationBase = buildAgentContinuationInput(
        apiInput,
        latestRound,
        useStore.getState().tasks,
        accumulatedOutputItems,
        toolCallsUsed,
        maxToolCalls,
      );
      // Insert function_call_output items before the continuation system message
      continuationBase.splice(
        continuationBase.length - 1,
        0,
        ...functionCallOutputs,
      );
      // Inject batch-generated images as input_image user message for model visibility
      const batchImagesItem = await createAgentBatchImagesInputItem(
        latestRound,
        useStore.getState().tasks,
        streamingTaskIds,
      );
      if (batchImagesItem)
        continuationBase.splice(
          continuationBase.length - 1,
          0,
          batchImagesItem,
        );
      apiInputForTurn = continuationBase;
      accumulatedOutputItems = accumulatedOutputItemsWithFunctionOutputs;
      pendingToolTextSeparator = true;
    }

    const taskIds: string[] = [...streamingTaskIds];
    const outputIds = taskIds.flatMap(
      (taskId) =>
        useStore.getState().tasks.find((task) => task.id === taskId)
          ?.outputImages ?? [],
    );
    const limitNotice = reachedToolLimit
      ? `已达到最大工具调用次数（${maxToolCalls}），已停止自动续跑。`
      : '';
    const joinedText = textSegments.join('\n\n').trim();
    const finalContent =
      [joinedText, limitNotice]
        .filter(Boolean)
        .join(joinedText ? '\n\n' : '') ||
      (taskIds.length > 0 || outputIds.length > 0 ? '图像已生成。' : '');

    const assistantMessage: AgentMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: finalContent,
      roundId,
      outputTaskIds: taskIds,
      createdAt: Date.now(),
    };

    updateAgentConversation(conversationId, (current) => ({
      ...current,
      updatedAt: Date.now(),
      rounds: current.rounds.map((round) =>
        round.id === roundId
          ? {
              ...round,
              assistantMessageId,
              outputTaskIds: taskIds,
              responseId: lastResponseId,
              responseOutput: accumulatedOutputItems,
              status: 'done',
              error: null,
              finishedAt: Date.now(),
            }
          : round,
      ),
      messages: current.messages.some(
        (message) => message.id === assistantMessageId,
      )
        ? current.messages.map((message) =>
            message.id === assistantMessageId ? assistantMessage : message,
          )
        : [...current.messages, assistantMessage],
    }));

    useStore
      .getState()
      .showToast(
        outputIds.length > 0 ? 'Agent 已生成图片' : 'Agent 已回复',
        'success',
      );
    showTaskCompletionNotification(
      outputIds.length > 0 ? 'Agent 已生成图片' : 'Agent 已回复',
      outputIds.length > 0
        ? `Agent 回复已结束，共生成 ${outputIds.length} 张图片。`
        : 'Agent 回复已结束。',
    );
  } catch (err) {
    if (controller.signal.aborted) {
      if (markAgentRoundStopped(conversationId, roundId)) {
        useStore.getState().showToast('已停止生成', 'info');
      }
      return;
    }

    const message = err instanceof Error ? err.message : String(err);

    updateAgentConversation(conversationId, (current) => {
      const failedRound = current.rounds.find((round) => round.id === roundId);
      const existingAssistantMessage = failedRound?.assistantMessageId
        ? current.messages.find(
            (item) => item.id === failedRound.assistantMessageId,
          )
        : current.messages.find(
            (item) => item.roundId === roundId && item.role === 'assistant',
          );
      const errorContent = `请求失败：${message}`;

      return {
        ...current,
        title:
          current.rounds.length === 1 && current.rounds[0].id === roundId
            ? '新对话'
            : current.title,
        updatedAt: Date.now(),
        rounds: current.rounds.map((round) =>
          round.id === roundId
            ? {
                ...round,
                ...(existingAssistantMessage
                  ? { assistantMessageId: existingAssistantMessage.id }
                  : {}),
                status: 'error',
                error: message,
                finishedAt: Date.now(),
              }
            : round,
        ),
        messages: existingAssistantMessage
          ? current.messages.map((item) =>
              item.id === existingAssistantMessage.id
                ? { ...item, content: errorContent }
                : item,
            )
          : [
              ...current.messages,
              {
                id: genId(),
                role: 'assistant',
                content: errorContent,
                roundId,
                createdAt: Date.now(),
              },
            ],
      };
    });
    useStore.getState().showToast(`Agent 请求失败：${message}`, 'error');
  } finally {
    if (agentRoundControllers.get(controllerKey) === controller) {
      agentRoundControllers.delete(controllerKey);
    }
  }
}

async function executeTask(taskId: string) {
  const { settings } = useStore.getState();
  const task = useStore.getState().tasks.find((t) => t.id === taskId);
  if (!task) return;
  if (!isAgentTask(task)) {
    await executeSystemImageTask(taskId);
    return;
  }
  const taskProfile = getTaskApiProfile(settings, task);
  if (!taskProfile && task.apiProfileId) {
    updateTaskInStore(taskId, {
      status: 'error',
      error: '找不到此任务所使用的 API 配置。',
      falRecoverable: false,
      customRecoverable: false,
      finishedAt: Date.now(),
      elapsed: Date.now() - task.createdAt,
    });
    return;
  }
  const activeProfile = taskProfile ?? getActiveApiProfile(settings);
  const requestSettings = createSettingsForApiProfile(settings, activeProfile);
  const taskProvider = task.apiProvider ?? activeProfile.provider;
  let falRequestInfo: { requestId: string; endpoint: string } | null =
    task.falRequestId && task.falEndpoint
      ? { requestId: task.falRequestId, endpoint: task.falEndpoint }
      : null;
  let customTaskInfo: { taskId: string } | null = task.customTaskId
    ? { taskId: task.customTaskId }
    : null;

  if (
    taskProvider !== 'fal' &&
    !isAsyncCustomProviderTask(
      requestSettings,
      taskProvider,
      task.inputImageIds.length > 0,
    )
  ) {
    scheduleOpenAIWatchdog(taskId, activeProfile.timeout, activeProfile);
  }

  try {
    // 获取输入图片 data URLs
    const inputDataUrls: string[] = [];
    for (const imgId of task.inputImageIds) {
      const dataUrl = await ensureImageCached(imgId);
      if (!dataUrl) throw new Error('输入图片已不存在');
      inputDataUrls.push(dataUrl);
    }
    let maskDataUrl: string | undefined;
    if (task.maskImageId) {
      maskDataUrl = await ensureImageCached(task.maskImageId);
      if (!maskDataUrl) throw new Error('遮罩图片已不存在');
    }

    const result = await callImageApi({
      settings: requestSettings,
      prompt: replaceImageMentionsForApi(task.prompt, inputDataUrls.length),
      params: task.params,
      inputImageDataUrls: inputDataUrls,
      maskDataUrl,
      onFalRequestEnqueued: (request) => {
        falRequestInfo = request;
        updateTaskInStore(taskId, {
          falRequestId: request.requestId,
          falEndpoint: request.endpoint,
          falRecoverable: false,
        });
      },
      onCustomTaskEnqueued: (request) => {
        customTaskInfo = request;
        updateTaskInStore(taskId, {
          customTaskId: request.taskId,
          customRecoverable: false,
        });
      },
      onPartialImage: (partial) => {
        useStore
          .getState()
          .setTaskStreamPreview(taskId, partial.image, partial.requestIndex);
        void persistTaskStreamPartialImage(
          taskId,
          partial.image,
          partial.requestIndex,
        );
      },
    });

    const latestBeforeSuccess = useStore
      .getState()
      .tasks.find((t) => t.id === taskId);
    if (!latestBeforeSuccess || latestBeforeSuccess.status !== 'running') {
      useStore.getState().setTaskStreamPreview(taskId);
      return;
    }

    // 存储输出图片
    const outputIds: string[] = [];
    for (const dataUrl of result.images) {
      const imgId = await storeImage(dataUrl, 'generated');
      cacheImage(imgId, dataUrl);
      outputIds.push(imgId);
    }
    const isAsyncCustomTask =
      taskProvider !== 'fal' &&
      taskProvider !== 'openai' &&
      Boolean(customTaskInfo);
    const actualParamsList =
      taskProvider === 'fal'
        ? await resolveImageSizeParamsList(
            result.images,
            result.actualParamsList,
          )
        : isAsyncCustomTask
          ? await readImageSizeParamsList(result.images)
          : result.actualParamsList;
    const actualParams = (() => {
      if (taskProvider === 'fal') return firstActualParams(actualParamsList);
      if (isAsyncCustomTask) return firstActualParams(actualParamsList);
      return { ...result.actualParams, n: outputIds.length };
    })();
    const shouldStoreRevisedPrompts =
      taskProvider !== 'fal' && !isAsyncCustomTask;
    const actualParamsByImage = mapActualParamsByImage(
      outputIds,
      actualParamsList,
    );
    const revisedPromptByImage = shouldStoreRevisedPrompts
      ? result.revisedPrompts?.reduce<Record<string, string>>(
          (acc, revisedPrompt, index) => {
            const imgId = outputIds[index];
            if (imgId && revisedPrompt && revisedPrompt.trim())
              acc[imgId] = revisedPrompt;
            return acc;
          },
          {},
        )
      : undefined;
    const promptWasRevised =
      shouldStoreRevisedPrompts &&
      result.revisedPrompts?.some(
        (revisedPrompt) =>
          revisedPrompt?.trim() && revisedPrompt.trim() !== task.prompt.trim(),
      );
    const hasRevisedPromptValue =
      shouldStoreRevisedPrompts &&
      result.revisedPrompts?.some((revisedPrompt) => revisedPrompt?.trim());
    if (
      taskProvider === 'openai' &&
      activeProfile.apiMode === 'responses' &&
      !activeProfile.codexCli
    ) {
      if (promptWasRevised) {
        showCodexCliPrompt();
      } else if (!hasRevisedPromptValue) {
        showCodexCliPrompt(false, '接口没有返回官方 API 会返回的部分信息');
      }
    }

    // 更新任务
    const latestBeforeUpdate = useStore
      .getState()
      .tasks.find((t) => t.id === taskId);
    if (!latestBeforeUpdate || latestBeforeUpdate.status !== 'running') {
      useStore.getState().setTaskStreamPreview(taskId);
      return;
    }
    const partialImageIdsToClean =
      latestBeforeUpdate.streamPartialImageIds || [];
    clearOpenAIWatchdogTimer(taskId);
    useStore.getState().setTaskStreamPreview(taskId);
    updateTaskInStore(taskId, {
      outputImages: outputIds,
      streamPartialImageIds: undefined,
      rawImageUrls: result.rawImageUrls?.length
        ? result.rawImageUrls
        : undefined,
      actualParams,
      actualParamsByImage,
      revisedPromptByImage:
        revisedPromptByImage && Object.keys(revisedPromptByImage).length > 0
          ? revisedPromptByImage
          : undefined,
      status: 'done',
      finishedAt: Date.now(),
      elapsed: Date.now() - task.createdAt,
      falRecoverable: false,
      customRecoverable: false,
    });
    void deleteUnreferencedImageIds(partialImageIdsToClean);

    useStore
      .getState()
      .showToast(`生成完成，共 ${outputIds.length} 张图片`, 'success');
    if (!isAgentTask(task))
      showTaskCompletionNotification(
        '图像生成完成',
        `生成完成，共 ${outputIds.length} 张图片。`,
      );
    const currentMask = useStore.getState().maskDraft;
    if (
      maskDataUrl &&
      currentMask &&
      currentMask.targetImageId === task.maskTargetImageId &&
      currentMask.maskDataUrl === maskDataUrl
    ) {
      useStore.getState().clearMaskDraft();
    }
  } catch (err) {
    clearOpenAIWatchdogTimer(taskId);
    const latestTask =
      useStore.getState().tasks.find((t) => t.id === taskId) ?? task;
    if (latestTask.status !== 'running') return;
    useStore.getState().setTaskStreamPreview(taskId);
    const latestFalRequestInfo =
      falRequestInfo ??
      (latestTask.falRequestId && latestTask.falEndpoint
        ? {
            requestId: latestTask.falRequestId,
            endpoint: latestTask.falEndpoint,
          }
        : null);
    const latestCustomTaskInfo =
      customTaskInfo ??
      (latestTask.customTaskId ? { taskId: latestTask.customTaskId } : null);
    if (
      latestTask.apiProvider === 'fal' &&
      latestFalRequestInfo &&
      isFalConnectionRecoverableError(err)
    ) {
      updateTaskInStore(taskId, {
        status: 'error',
        error: '与 fal.ai 的连接已断开，之后会继续查询任务结果。',
        falRequestId: latestFalRequestInfo.requestId,
        falEndpoint: latestFalRequestInfo.endpoint,
        falRecoverable: true,
        finishedAt: Date.now(),
        elapsed: Date.now() - task.createdAt,
      });
      scheduleFalRecovery(taskId);
    } else if (latestCustomTaskInfo && isFalConnectionRecoverableError(err)) {
      updateTaskInStore(taskId, {
        status: 'error',
        error: '与自定义异步任务的连接已断开，之后会继续查询任务结果。',
        customTaskId: latestCustomTaskInfo.taskId,
        customRecoverable: true,
        finishedAt: Date.now(),
        elapsed: Date.now() - task.createdAt,
      });
      scheduleCustomRecovery(taskId);
    } else {
      let errorMessage = getTaskErrorMessage(err);
      const settings = useStore.getState().settings;
      const profile = getTaskApiProfile(settings, latestTask);
      const usesApiProxy = profile?.apiProxy ?? settings.apiProxy;
      const activeProfile = getActiveApiProfile(settings);
      const hintProfile = profile ?? {
        provider: latestTask.apiProvider ?? activeProfile.provider,
        apiMode: settings.apiMode,
        streamImages: activeProfile.streamImages,
        streamPartialImages: activeProfile.streamPartialImages,
      };
      const networkErrorHint = getApiRequestNetworkErrorHint(
        err,
        latestTask.createdAt,
        usesApiProxy,
        hintProfile,
      );
      if (networkErrorHint && !errorMessage.includes(IMAGE_FETCH_CORS_HINT)) {
        errorMessage += `\n${networkErrorHint}`;
      }
      updateTaskInStore(taskId, {
        status: 'error',
        error: errorMessage,
        ...getRawErrorPayload(err),
        falRecoverable: false,
        customRecoverable: false,
        finishedAt: Date.now(),
        elapsed: Date.now() - task.createdAt,
      });
      useStore.getState().setDetailTaskId(taskId);
    }
  } finally {
    // 释放输入图片的内存缓存（已持久化到 IndexedDB，后续按需从 DB 加载）
    for (const imgId of task.inputImageIds) {
      imageCache.delete(imgId);
    }
  }
}

function normalizeFavoritePatch(
  task: TaskRecord,
  patch: Partial<TaskRecord>,
  defaultFavoriteCollectionId: string | null,
): Partial<TaskRecord> {
  if ('favoriteCollectionIds' in patch) {
    const ids = normalizeFavoriteCollectionIds(patch.favoriteCollectionIds);
    return { ...patch, favoriteCollectionIds: ids, isFavorite: ids.length > 0 };
  }
  if ('isFavorite' in patch) {
    if (patch.isFavorite) {
      const ids = normalizeFavoriteCollectionIds(task.favoriteCollectionIds);
      return {
        ...patch,
        favoriteCollectionIds: ids.length
          ? ids
          : defaultFavoriteCollectionId
            ? [defaultFavoriteCollectionId]
            : [],
      };
    }
    return { ...patch, favoriteCollectionIds: [] };
  }
  return patch;
}

export function updateTaskInStore(taskId: string, patch: Partial<TaskRecord>) {
  const { tasks, setTasks, defaultFavoriteCollectionId } = useStore.getState();
  const updated = tasks.map((t) =>
    t.id === taskId
      ? {
          ...t,
          ...normalizeFavoritePatch(t, patch, defaultFavoriteCollectionId),
        }
      : t,
  );
  const task = updated.find((t) => t.id === taskId);
  setTasks(updated);
  maybeOpenSupportPrompt(tasks, updated, taskId);
  if (task) putTask(task);
}

function normalizeFavoriteCollectionIds(ids: unknown) {
  if (!Array.isArray(ids)) return [];
  return Array.from(
    new Set(
      ids.map(String).filter((id) => id && id !== ALL_FAVORITES_COLLECTION_ID),
    ),
  );
}

function sameFavoriteCollectionIds(a: string[], b: string[]) {
  if (a.length !== b.length) return false;
  const bSet = new Set(b);
  return a.every((id) => bSet.has(id));
}

export function getTaskFavoriteCollectionIds(task: TaskRecord) {
  const ids = normalizeFavoriteCollectionIds(task.favoriteCollectionIds);
  if (ids.length > 0) return ids;
  const defaultFavoriteCollectionId =
    useStore.getState().defaultFavoriteCollectionId;
  return task.isFavorite && defaultFavoriteCollectionId
    ? [defaultFavoriteCollectionId]
    : [];
}

function normalizeTaskFavoriteState(
  task: TaskRecord,
  collections: FavoriteCollection[],
): TaskRecord {
  const collectionIdSet = new Set(
    collections.map((collection) => collection.id),
  );
  const normalizedIds = normalizeFavoriteCollectionIds(
    task.favoriteCollectionIds,
  ).filter((id) => collectionIdSet.has(id));
  // 旧版本只有 isFavorite 没有 favoriteCollectionIds，迁移到"默认"收藏夹
  const defaultId = getDefaultNamedFavoriteCollectionId(collections);
  const ids =
    normalizedIds.length > 0
      ? normalizedIds
      : task.isFavorite && defaultId
        ? [defaultId]
        : [];
  const isFavorite = ids.length > 0 || Boolean(task.isFavorite);
  if (
    ids.length === (task.favoriteCollectionIds ?? []).length &&
    ids.every((id, index) => id === task.favoriteCollectionIds?.[index]) &&
    Boolean(task.isFavorite) === isFavorite
  ) {
    return task;
  }
  return { ...task, favoriteCollectionIds: ids, isFavorite };
}

function normalizeLoadedFavoriteState(
  tasks: TaskRecord[],
  collections: FavoriteCollection[],
  preferredDefaultFavoriteCollectionId: string | null,
) {
  let changed = false;
  // 确保"默认"收藏夹存在，给孤立收藏任务一个归属
  const normalizedCollections = ensureDefaultNamedCollection(
    ensureDefaultFavoriteCollection(normalizeFavoriteCollections(collections)),
  );
  const defaultFavoriteCollectionId = resolveDefaultFavoriteCollectionId(
    normalizedCollections,
    preferredDefaultFavoriteCollectionId,
  );
  const normalizedTasks = tasks.map((task) => {
    const nextTask = normalizeTaskFavoriteState(task, normalizedCollections);
    if (nextTask !== task) changed = true;
    return nextTask;
  });
  return {
    tasks: normalizedTasks,
    collections: normalizedCollections,
    defaultFavoriteCollectionId,
    changed,
  };
}

export function getFavoriteCollectionTitle(
  collectionId: string | null,
  collections = useStore.getState().favoriteCollections,
) {
  if (collectionId === ALL_FAVORITES_COLLECTION_ID) return '全部';
  return (
    collections.find((collection) => collection.id === collectionId)?.name ??
    DEFAULT_FAVORITE_COLLECTION_NAME
  );
}

export function createFavoriteCollection(name: string) {
  const normalizedName = normalizeFavoriteCollectionName(name);
  if (!normalizedName) return null;
  if (Array.from(normalizedName).length > 60) {
    useStore.getState().showToast('收藏夹名称最多 60 个字符', 'error');
    return null;
  }
  const state = useStore.getState();
  const existing = state.favoriteCollections.find(
    (collection) => collection.name === normalizedName,
  );
  if (existing) return existing;
  const now = Date.now();
  const collection: FavoriteCollection = {
    id: genId(),
    name: normalizedName,
    createdAt: now,
    updatedAt: now,
  };
  state.setFavoriteCollections([...state.favoriteCollections, collection]);
  state.showToast(`已创建收藏夹「${normalizedName}」`, 'success');
  return collection;
}

export function renameFavoriteCollection(collectionId: string, name: string) {
  const normalizedName = normalizeFavoriteCollectionName(name);
  if (!normalizedName || collectionId === ALL_FAVORITES_COLLECTION_ID) return;
  if (Array.from(normalizedName).length > 60) {
    useStore.getState().showToast('收藏夹名称最多 60 个字符', 'error');
    return;
  }
  const { favoriteCollections, setFavoriteCollections, showToast } =
    useStore.getState();
  setFavoriteCollections(
    favoriteCollections.map((collection) =>
      collection.id === collectionId
        ? { ...collection, name: normalizedName, updatedAt: Date.now() }
        : collection,
    ),
  );
  showToast('收藏夹名称已更新', 'success');
}

export async function updateTasksFavoriteCollections(
  taskIds: string[],
  collectionIds: string[],
) {
  const ids = normalizeFavoriteCollectionIds(collectionIds);
  const uniqueTaskIds = Array.from(new Set(taskIds)).filter(Boolean);
  if (!uniqueTaskIds.length) return;
  const { tasks, setTasks, clearSelection, showToast } = useStore.getState();
  const idSet = new Set(uniqueTaskIds);
  const changedTaskIds = new Set<string>();
  const updated = tasks.map((task) => {
    if (!idSet.has(task.id)) return task;
    if (sameFavoriteCollectionIds(getTaskFavoriteCollectionIds(task), ids))
      return task;
    changedTaskIds.add(task.id);
    return { ...task, favoriteCollectionIds: ids, isFavorite: ids.length > 0 };
  });
  if (!changedTaskIds.size) {
    clearSelection();
    return;
  }
  setTasks(updated);
  await Promise.all(
    updated
      .filter((task) => changedTaskIds.has(task.id))
      .map((task) => putTask(task)),
  );
  clearSelection();
  showToast(ids.length ? '收藏夹已更新' : '已取消收藏', 'success');
}

export async function deleteFavoriteCollection(
  collectionId: string,
  deleteTasks = false,
) {
  if (!collectionId || collectionId === ALL_FAVORITES_COLLECTION_ID) return;
  const state = useStore.getState();
  const collection = state.favoriteCollections.find(
    (item) => item.id === collectionId,
  );
  if (!collection || state.favoriteCollections.length <= 1) return;
  const collectionTaskRefs = state.tasks
    .map((task) => ({ task, favoriteIds: getTaskFavoriteCollectionIds(task) }))
    .filter(({ favoriteIds }) => favoriteIds.includes(collectionId));
  const taskIds = collectionTaskRefs.map(({ task }) => task.id);
  const nextCollections = state.favoriteCollections.filter(
    (item) => item.id !== collectionId,
  );
  const nextCollectionIdSet = new Set(nextCollections.map((item) => item.id));
  state.setFavoriteCollections(nextCollections);
  if (state.defaultFavoriteCollectionId === collectionId) {
    const nextDefaultId = nextCollections[0]?.id;
    if (nextDefaultId)
      useStore.getState().setDefaultFavoriteCollectionId(nextDefaultId);
  }
  if (state.activeFavoriteCollectionId === collectionId)
    state.setActiveFavoriteCollectionId(null);
  if (deleteTasks) {
    const idsByTaskToKeep = new Map<string, string[]>();
    const taskIdsToDelete: string[] = [];
    for (const { task, favoriteIds } of collectionTaskRefs) {
      const nextIds = favoriteIds.filter(
        (id) => id !== collectionId && nextCollectionIdSet.has(id),
      );
      if (nextIds.length) {
        idsByTaskToKeep.set(task.id, nextIds);
      } else {
        taskIdsToDelete.push(task.id);
      }
    }
    if (idsByTaskToKeep.size) {
      const latestTasks = useStore.getState().tasks;
      const updated = latestTasks.map((task) => {
        const ids = idsByTaskToKeep.get(task.id);
        return ids
          ? { ...task, favoriteCollectionIds: ids, isFavorite: true }
          : task;
      });
      useStore.getState().setTasks(updated);
      await Promise.all(
        updated
          .filter((task) => idsByTaskToKeep.has(task.id))
          .map((task) => putTask(task)),
      );
    }
    if (taskIdsToDelete.length) await removeMultipleTasks(taskIdsToDelete);
  } else if (taskIds.length) {
    const idsByTaskId = new Map(
      collectionTaskRefs.map(({ task, favoriteIds }) => [
        task.id,
        favoriteIds.filter(
          (id) => id !== collectionId && nextCollectionIdSet.has(id),
        ),
      ]),
    );
    const updated = state.tasks.map((task) => {
      const ids = idsByTaskId.get(task.id);
      if (!ids) return task;
      return {
        ...task,
        favoriteCollectionIds: ids,
        isFavorite: ids.length > 0,
      };
    });
    state.setTasks(updated);
    await Promise.all(
      updated
        .filter((task) => idsByTaskId.has(task.id))
        .map((task) => putTask(task)),
    );
  }
  useStore
    .getState()
    .setSelectedFavoriteCollectionIds((ids) =>
      ids.filter((id) => id !== collectionId),
    );
  useStore
    .getState()
    .showToast(`已删除收藏夹「${collection.name}」`, 'success');
}

/** 重试失败的任务：创建新任务并替换原失败任务 */
export async function retryTask(task: TaskRecord) {
  const { settings } = useStore.getState();
  const state = useStore.getState();
  const isGalleryTask = !isAgentTask(task);
  const systemModel = isGalleryTask
    ? state.systemImageModels.find(
        (model) => model.site_model_id === task.apiModel,
      ) || null
    : null;
  const taskProfile = isGalleryTask ? null : getTaskApiProfile(settings, task);
  const fallbackProfile =
    !isGalleryTask && !task.apiProfileId ? getActiveApiProfile(settings) : null;
  const profileForParams = taskProfile ?? fallbackProfile;
  const paramsSettings = profileForParams
    ? createSettingsForApiProfile(settings, profileForParams)
    : settings;
  const normalizedParams = normalizeParamsForSystemImageModel(
    normalizeParamsForSettings(task.params, paramsSettings, {
      hasInputImages: task.inputImageIds.length > 0,
    }),
    systemModel,
  );
  const taskId = genId();
  const apiProvider = isGalleryTask
    ? task.apiProvider || systemModel?.provider_name || 'system'
    : task.apiProvider || profileForParams?.provider || '';
  const apiProfileId = isGalleryTask
    ? ''
    : task.apiProfileId || profileForParams?.id || '';
  const apiProfileName = isGalleryTask
    ? task.apiProfileName ||
      systemModel?.display_name ||
      task.apiModel ||
      '系统图片模型'
    : task.apiProfileName ||
      profileForParams?.name ||
      task.apiModel ||
      '未知配置';
  const apiMode = isGalleryTask
    ? task.apiMode || systemModel?.api_mode || 'images'
    : task.apiMode || profileForParams?.apiMode || 'images';
  const apiModel = isGalleryTask
    ? task.apiModel || systemModel?.site_model_id || ''
    : task.apiModel || profileForParams?.model || '';
  const newTask: TaskRecord = {
    id: taskId,
    prompt: task.prompt,
    params: normalizedParams,
    apiProvider,
    apiProfileId,
    apiProfileName,
    apiMode,
    apiModel,
    streamPartialImages: task.streamPartialImages,
    inputImageIds: [...task.inputImageIds],
    maskTargetImageId: task.maskTargetImageId ?? null,
    maskImageId: task.maskImageId ?? null,
    outputImages: [],
    status: 'running',
    error: null,
    createdAt: Date.now(),
    finishedAt: null,
    elapsed: null,
    isFavorite: task.isFavorite,
    favoriteCollectionIds: task.favoriteCollectionIds
      ? [...task.favoriteCollectionIds]
      : undefined,
    sourceMode: isGalleryTask ? 'gallery' : task.sourceMode,
    agentConversationId: task.agentConversationId,
    agentRoundId: task.agentRoundId,
    agentMessageId: task.agentMessageId,
    agentToolCallId: task.agentToolCallId,
    agentBatchCallId: task.agentBatchCallId,
    agentToolAction: task.agentToolAction,
  };

  const latestTasks = useStore.getState().tasks;
  const taskIndex = latestTasks.findIndex((item) => item.id === task.id);
  const nextTasks =
    taskIndex >= 0
      ? latestTasks.map((item) => (item.id === task.id ? newTask : item))
      : [newTask, ...latestTasks];
  useStore.getState().setTasks(nextTasks);
  if (taskIndex >= 0) {
    useStore
      .getState()
      .setSelectedTaskIds((ids) =>
        ids.map((id) => (id === task.id ? newTask.id : id)),
      );
    await dbDeleteTask(task.id);
  }
  if ((task.sourceMode ?? 'gallery') === 'gallery') {
    const generationId = getSystemGenerationIdForDelete(task);
    if (generationId) hiddenSystemGenerationIds.add(generationId);
    try {
      await deleteRemoteImageGenerations([task]);
    } catch (err) {
      console.warn('Failed to delete retried image generation', err);
    } finally {
      if (generationId) hiddenSystemGenerationIds.delete(generationId);
    }
  }
  await putTask(newTask);

  executeTask(taskId);
}

/** 复用配置 */
export async function reuseConfig(task: TaskRecord) {
  const {
    settings,
    setPrompt,
    setParams,
    setInputImages,
    setMaskDraft,
    clearMaskDraft,
    showToast,
    setConfirmDialog,
    setReusedTaskApiProfile,
  } = useStore.getState();
  const normalizedSettings = normalizeSettings(settings);
  const currentProfile = getActiveApiProfile(settings);
  const matchedProfile = normalizedSettings.reuseTaskApiProfileTemporarily
    ? getTaskApiProfile(normalizedSettings, task)
    : null;
  const shouldTemporarilyReuseProfile = Boolean(
    matchedProfile && matchedProfile.id !== currentProfile.id,
  );
  const missingReusedProfile =
    normalizedSettings.reuseTaskApiProfileTemporarily && !matchedProfile;
  const taskProfileName = matchedProfile?.name ?? getTaskApiProfileName(task);
  const paramsSettings =
    shouldTemporarilyReuseProfile && matchedProfile
      ? createSettingsForApiProfile(normalizedSettings, matchedProfile)
      : normalizedSettings;

  setParams(
    normalizeParamsForSettings(task.params, paramsSettings, {
      hasInputImages: task.inputImageIds.length > 0,
    }),
  );
  setReusedTaskApiProfile(
    shouldTemporarilyReuseProfile && matchedProfile ? matchedProfile.id : null,
    missingReusedProfile,
    taskProfileName,
  );
  clearMaskDraft();

  // 恢复输入图片
  const imgs: InputImage[] = [];
  for (const imgId of task.inputImageIds) {
    const dataUrl = await ensureImageCached(imgId);
    if (dataUrl) {
      imgs.push({ id: imgId, dataUrl });
    }
  }
  setInputImages(imgs);
  setPrompt(task.prompt);
  const maskTargetImageId =
    task.maskTargetImageId ?? (task.maskImageId ? task.inputImageIds[0] : null);
  if (
    maskTargetImageId &&
    task.maskImageId &&
    imgs.some((img) => img.id === maskTargetImageId)
  ) {
    const maskDataUrl = await ensureImageCached(task.maskImageId);
    if (maskDataUrl) {
      setMaskDraft({
        targetImageId: maskTargetImageId,
        maskDataUrl,
        updatedAt: Date.now(),
      });
    } else {
      clearMaskDraft();
    }
  } else {
    clearMaskDraft();
  }
  if (missingReusedProfile) {
    setConfirmDialog({
      title: '找不到 API 配置',
      message: `找不到复用任务所使用的 API 配置「${taskProfileName}」，要使用当前的 API 配置「${currentProfile.name}」提交任务吗？`,
      confirmText: '使用当前配置提交',
      cancelText: '放弃提交',
      action: () => {
        void submitTask({ useCurrentApiProfileWhenReusedMissing: true });
      },
    });
    return;
  }

  showToast(
    shouldTemporarilyReuseProfile && matchedProfile
      ? `已临时复用该任务的 API 配置「${matchedProfile.name}」`
      : '已复用配置到输入框',
    'success',
  );
}

/** 编辑输出：将输出图加入输入 */
export async function editOutputs(task: TaskRecord) {
  const { inputImages, addInputImage, showToast } = useStore.getState();
  const outputImages = task.outputImages ?? [];
  const rawImageUrls = task.rawImageUrls ?? [];
  const outputCount = Math.max(outputImages.length, rawImageUrls.length);
  if (outputCount === 0) return;

  let added = 0;
  let failed = 0;
  const existingInputIds = new Set(inputImages.map((image) => image.id));
  const resolvedOutputImages = [...outputImages];
  for (let index = 0; index < outputCount; index++) {
    try {
      let imgId = resolvedOutputImages[index] || '';
      let dataUrl = imgId ? await ensureImageCached(imgId) : '';
      if (!dataUrl && rawImageUrls[index]) {
        dataUrl = await imageUrlToDataUrl(rawImageUrls[index]);
        imgId = await storeImage(dataUrl, 'generated');
        cacheImage(imgId, dataUrl);
        resolvedOutputImages[index] = imgId;
      }
      if (!imgId || !dataUrl) {
        failed++;
        continue;
      }
      if (existingInputIds.has(imgId)) continue;
      addInputImage({ id: imgId, dataUrl });
      existingInputIds.add(imgId);
      added++;
    } catch (err) {
      console.error(err);
      failed++;
    }
  }
  const nextOutputImages = resolvedOutputImages.filter(Boolean);
  if (
    nextOutputImages.length > 0 &&
    nextOutputImages.join('\n') !== outputImages.join('\n')
  ) {
    updateTaskInStore(task.id, { outputImages: nextOutputImages });
  }
  showToast(
    failed > 0
      ? `已添加 ${added} 张输出图到输入，${failed} 张读取失败`
      : `已添加 ${added} 张输出图到输入`,
    failed > 0 && added === 0 ? 'error' : 'success',
  );
}

/** 删除多条任务 */
export async function removeMultipleTasks(taskIds: string[]) {
  const {
    tasks,
    setTasks,
    inputImages,
    galleryInputDraft,
    showToast,
    clearSelection,
    selectedTaskIds,
  } = useStore.getState();

  if (!taskIds.length) return;

  const toDelete = new Set(taskIds);
  const deletedTasks = tasks.filter((t) => toDelete.has(t.id));
  let remoteDeleteFailed = false;
  try {
    await deleteRemoteImageGenerations(deletedTasks);
  } catch {
    remoteDeleteFailed = true;
  }
  const remaining = await scrubAgentOutputPayloadsForDeletedTasks(
    deletedTasks,
    tasks.filter((t) => !toDelete.has(t.id)),
  );

  // 收集所有被删除任务的关联图片
  const deletedImageIds = new Set<string>();
  for (const t of tasks) {
    if (toDelete.has(t.id)) {
      addTaskReferencedImageIds(deletedImageIds, t);
    }
  }

  setTasks(remaining);
  for (const id of taskIds) {
    await dbDeleteTask(id);
  }

  // 找出其他任务仍引用的图片
  const stillUsed = new Set<string>();
  for (const t of remaining) {
    addTaskReferencedImageIds(stillUsed, t);
  }
  addAgentReferencedImageIds(stillUsed);
  addInputDraftReferencedImageIds(stillUsed, galleryInputDraft);
  for (const img of inputImages) stillUsed.add(img.id);

  // 删除孤立图片
  for (const imgId of deletedImageIds) {
    if (!stillUsed.has(imgId)) {
      await deleteImage(imgId);
      imageCache.delete(imgId);
      thumbnailCache.delete(imgId);
    }
  }

  // 如果删除的任务在选中列表中，则移除
  const newSelection = selectedTaskIds.filter((id) => !toDelete.has(id));
  if (newSelection.length !== selectedTaskIds.length) {
    useStore.getState().setSelectedTaskIds(newSelection);
  }

  showToast(
    remoteDeleteFailed
      ? `已删除 ${taskIds.length} 个本地任务，但部分后端记录删除失败，刷新后可能重新出现`
      : `已删除 ${taskIds.length} 个任务`,
    remoteDeleteFailed ? 'error' : 'success',
  );
}

/** 删除单条任务 */
export async function removeTask(task: TaskRecord) {
  const { tasks, setTasks, inputImages, galleryInputDraft, showToast } =
    useStore.getState();
  let remoteDeleteFailed = false;
  try {
    await deleteRemoteImageGenerations([task]);
  } catch {
    remoteDeleteFailed = true;
  }

  // 收集此任务关联的图片
  const taskImageIds = new Set([
    ...(task.inputImageIds || []),
    ...(task.maskImageId ? [task.maskImageId] : []),
    ...(task.outputImages || []),
    ...(task.streamPartialImageIds || []),
  ]);

  // 从列表移除
  const remaining = await scrubAgentOutputPayloadsForDeletedTasks(
    [task],
    tasks.filter((t) => t.id !== task.id),
  );
  setTasks(remaining);
  await dbDeleteTask(task.id);

  // 找出其他任务仍引用的图片
  const stillUsed = new Set<string>();
  for (const t of remaining) {
    addTaskReferencedImageIds(stillUsed, t);
  }
  addAgentReferencedImageIds(stillUsed);
  addInputDraftReferencedImageIds(stillUsed, galleryInputDraft);
  for (const img of inputImages) stillUsed.add(img.id);

  // 删除孤立图片
  for (const imgId of taskImageIds) {
    if (!stillUsed.has(imgId)) {
      await deleteImage(imgId);
      imageCache.delete(imgId);
      thumbnailCache.delete(imgId);
    }
  }

  showToast(
    remoteDeleteFailed
      ? '本地任务已删除，但后端记录删除失败，刷新后可能重新出现'
      : '任务已删除',
    remoteDeleteFailed ? 'error' : 'success',
  );
}

/** 添加图片到输入（文件上传） */
export async function addImageFromFile(file: File): Promise<void> {
  const image = await createInputImageFromFile(file);
  if (!image) return;
  useStore.getState().addInputImage(image);
}

export async function createInputImageFromFile(
  file: File,
): Promise<InputImage | null> {
  if (!file.type.startsWith('image/')) return null;
  const dataUrl = await fileToDataUrl(file);
  const dimensions = await getImageDimensions(dataUrl);
  const id = await storeImage(dataUrl, 'upload');
  cacheImage(id, dataUrl);
  return { id, dataUrl, ...dimensions };
}

/** 添加图片到输入（右键菜单）—— 支持 data/blob/http URL */
export async function addImageFromUrl(src: string): Promise<void> {
  const res = await fetch(src);
  const blob = await res.blob();
  if (!blob.type.startsWith('image/')) throw new Error('不是有效的图片');
  const dataUrl = await blobToDataUrl(blob);
  const dimensions = await getImageDimensions(dataUrl);
  const id = await storeImage(dataUrl, 'upload');
  cacheImage(id, dataUrl);
  useStore.getState().addInputImage({ id, dataUrl, ...dimensions });
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
