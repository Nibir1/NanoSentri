import { describe, it, expect, vi, type Mocked, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import App from '../App';

// Mock Axios
vi.mock('axios');
const mockedAxios = axios as Mocked<typeof axios>;

describe('NanoSentri Dashboard', () => {

    // IMPORTANT: Clear mocks between tests to prevent data pollution
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders correctly and checks health', async () => {
        mockedAxios.get.mockImplementation((url: string) => {
            if (url.includes('/health')) return Promise.resolve({ data: { status: 'online' } });
            if (url.includes('/benchmark')) return Promise.resolve({ data: { status: 'no_data' } });
            return Promise.reject(new Error('not found'));
        });

        render(<App />);

        expect(screen.getByText(/NanoSentri/i)).toBeInTheDocument();

        // Wait for the "loading" state to flip to "online"
        await waitFor(() => {
            expect(screen.getByText('online')).toBeInTheDocument();
        });
    });

    it('runs diagnostics when button is clicked', async () => {
        // 1. Setup All Mocks (GET for health, POST for diagnose)
        mockedAxios.get.mockImplementation((url: string) => {
            if (url.includes('/health')) return Promise.resolve({ data: { status: 'online' } });
            if (url.includes('/benchmark')) return Promise.resolve({ data: { status: 'no_data' } });
            return Promise.resolve({});
        });

        mockedAxios.post.mockResolvedValue({
            data: {
                diagnosis: "CRITICAL FAILURE detected.",
                inference_time_sec: 0.5,
                model_version: "test-v1"
            }
        });

        render(<App />);

        // --- CRITICAL FIX START ---
        // Wait for the app to become ONLINE. 
        // Before this happens, the "Run Diagnostics" button is disabled.
        await waitFor(() => {
            expect(screen.getByText('online')).toBeInTheDocument();
        });
        // --- CRITICAL FIX END ---

        // 2. Fill inputs
        const input = screen.getByPlaceholderText(/e.g. Why is the wind sensor/i);
        fireEvent.change(input, { target: { value: 'Test Query' } });

        // 3. Click Button (Now enabled because we waited for online status)
        const button = screen.getByText(/Run Diagnostics/i);
        fireEvent.click(button);

        // 4. Expect "Processing" state
        // Use waitFor to catch the UI update
        await waitFor(() => {
            expect(screen.getByText(/Processing/i)).toBeInTheDocument();
        });

        // 5. Wait for Result
        await waitFor(() => {
            expect(screen.getByText(/CRITICAL FAILURE/i)).toBeInTheDocument();
        });
    });
});